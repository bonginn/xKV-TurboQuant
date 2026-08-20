import torch
from transformers.cache_utils import DynamicCache
import gc
from pathlib import Path

from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
)

from ..configurations import xKVConfig
from ..runtime_audit import (
    record_cache_created,
    record_decode_update,
    record_group_merge,
    record_pair_call,
    record_reconstruction_check,
)
from .quant import Quantizer
from .turboquant import TurboQuantMSE, load_codebook

def fake_svd(
    tensor,
    rank,
    apply_hadamard_transform=False,
    quantizer=None,
    runtime_audit=None,
    audit_side=None,
):
    """Perform fake SVD: SVD -> Truncate -> (Optional Hadamard + Quantization) -> Multiply back."""
    bs, nh, sl, hd = tensor.shape
    tensor_reshaped = tensor.transpose(1, 2).reshape(bs, sl, nh * hd)

    is_turboquant = isinstance(quantizer, TurboQuantMSE)
    if is_turboquant and apply_hadamard_transform:
        raise ValueError(
            "TurboQuant-MSE already applies its own Haar rotation and cannot "
            "be combined with the xKV Hadamard transform"
        )
    if is_turboquant and quantizer.dimension != rank:
        raise ValueError(
            f"TurboQuant dimension {quantizer.dimension} must match the "
            f"requested SVD rank {rank}"
        )
    
    # Step 1: Perform SVD NOTE(brian1009): Have deterministic issue but faster
    #U_trunc, S_trunc, V_trunc = torch.svd_lowrank(tensor_reshaped, q=rank)
    #Vt_trunc = V_trunc.transpose(1, 2)
    
    U, S, V_h = torch.linalg.svd(tensor_reshaped, full_matrices=False)
    U_trunc = U[:, :, :rank]
    S_trunc = S[:, :rank]
    Vt_trunc = V_h[:, :rank, :]
    
    # Step 2: Split singular values evenly between U and V
    sqrt_S = torch.sqrt(S_trunc)  # Square root for even distribution
    U_scaled = U_trunc * sqrt_S.unsqueeze(1)  # Scale U with sqrt(S): (bs, sl, rank) * (bs, 1, rank)
    Vt_scaled = sqrt_S.unsqueeze(-1) * Vt_trunc  # Scale Vt with sqrt(S): (bs, rank, 1) * (bs, rank, nh*hd)

    # A short prompt (or narrow matrix) can have an effective SVD rank below
    # the configured rank.  Zero-padding both factors preserves their product
    # exactly while retaining the nominal latent dimension/codebook expected by
    # TurboQuant.
    effective_rank = U_scaled.shape[-1]
    if is_turboquant and effective_rank < rank:
        missing_rank = rank - effective_rank
        U_scaled = torch.cat(
            [U_scaled, U_scaled.new_zeros(bs, sl, missing_rank)],
            dim=-1,
        )
        Vt_scaled = torch.cat(
            [
                Vt_scaled,
                Vt_scaled.new_zeros(bs, missing_rank, nh * hd),
            ],
            dim=-2,
        )
    
    # Step 2.5: TurboQuant jointly transforms the low-rank factor pair.  The
    # inverse rotation is folded into Vt, so it must run before the legacy
    # Hadamard/RTN path and never stack another transform.
    if is_turboquant:
        factor_pair = quantizer.fake_quantize_pair(U_scaled, Vt_scaled)
        record_pair_call(
            runtime_audit,
            audit_side,
            quantizer,
            factor_pair,
        )
        U_final = factor_pair.left_factor
        Vt_final = factor_pair.right_factor
    elif apply_hadamard_transform:
        # Apply Hadamard transform directly to batch tensors (no need to split)
        # Reshape to 2D for Hadamard transform: (bs, sl, rank) -> (bs*sl, rank)
        U_reshaped = U_scaled.reshape(bs * sl, rank)
        # Reshape to 2D for Hadamard transform: (bs, rank, nh*hd) -> (bs*rank, nh*hd)
        Vt_reshaped = Vt_scaled.reshape(bs * rank, nh * hd)
        
        # Apply Hadamard transform
        from .hadamard_utils import apply_hadamard
        U_hadamard = apply_hadamard(U_reshaped)
        Vt_hadamard = apply_hadamard(Vt_reshaped.t()).t()  # B @ Q^T = (Q @ B^T)^T
        
        # Reshape back to original batch format
        U_hadamard = U_hadamard.reshape(bs, sl, rank)
        Vt_hadamard = Vt_hadamard.reshape(bs, rank, nh * hd)
        
        U_processed = U_hadamard
        Vt_processed = Vt_hadamard
    elif not is_turboquant:
        U_processed = U_scaled
        Vt_processed = Vt_scaled

    # Apply legacy RTN fake quantization if provided.  Only the left factor is
    # quantized in the original xKV implementation.
    if not is_turboquant:
        if quantizer is not None:
            U_final = quantizer(U_processed)
            Vt_final = Vt_processed
        else:
            U_final = U_processed
            Vt_final = Vt_processed
    
    # Step 3: Multiply back to approximate the original tensor.  Keep the
    # TurboQuant accuracy reference in FP32 even when model.generate() is
    # wrapped by an outer BF16/FP16 autocast context.
    if is_turboquant:
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            approx_tensor = torch.matmul(U_final, Vt_final)
    else:
        approx_tensor = torch.matmul(U_final, Vt_final)
    
    # # Step 2: Multiply back to approximate the original tensor
    # approx_tensor = torch.matmul(U_trunc, S_trunc.unsqueeze(-1) * Vt_trunc) # (bs, sl, nh * hd)
    approx_tensor = approx_tensor.view(bs, sl, nh, hd).transpose(1, 2)

    if is_turboquant and runtime_audit is not None:
        is_finite = bool(torch.isfinite(approx_tensor).all().item())
        record_reconstruction_check(
            runtime_audit,
            is_finite=is_finite,
        )
        if not is_finite:
            raise RuntimeError(
                f"TurboQuant {audit_side.upper()} reconstruction contains "
                "non-finite values"
            )
    
    return approx_tensor


def slerp_merge_rows_batch(X1, X2, t=0.5, gamma=0.05):
    """
    Vectorized row-wise SLERP merge of X1, X2 in shape (L, d), returning E of shape (L, d).
    - If a row is zero-norm in either X1 or X2, or if the angle is extremely small,
    we revert to linear interpolation to avoid numerical issues.

    The SLERP formula per row i is:
        e[i] = sin((1-t)*Omega[i]) / sin(Omega[i]) * (x1[i]/||x1[i]||)
            + sin(t*Omega[i])     / sin(Omega[i]) * (x2[i]/||x2[i]||),
    where Omega[i] = arccos( (x1[i] dot x2[i]) / (||x1[i]|| ||x2[i]||) ).
    """

    
    
    # 1. Compute row-wise norms => shape (L,1)
    norm1 = X1.norm(dim=1, keepdim=True)
    norm2 = X2.norm(dim=1, keepdim=True)


    # 3. Normalized vectors
    u1 = X1 / norm1
    u2 = X2 / norm2

    # 4. Dot product => shape (L,1)
    dot_val = (u1 * u2).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

    # 5. Angle and sine
    Omega = torch.acos(dot_val)  # shape (L,1)
    sinOmega = torch.sin(Omega)

    # s6. Divergence threshold
    d_min = Omega.min()
    d_max = Omega.max()
    threshold = d_min + (d_max - d_min) * gamma
    diverge_mask = Omega > threshold
    
    # near-parallel => angle < eps => linear fallback
    parallel_mask = (Omega < 1e-7)

    # 6. Compute SLERP coefficients
    #    alpha[i] = sin((1-t)*Omega[i]) / sin(Omega[i])
    #    beta[i]  = sin(t*Omega[i])     / sin(Omega[i])
    alpha = torch.sin((1.0 - t) * Omega) / sinOmega
    beta  = torch.sin(t * Omega)         / sinOmega

    # 7. SLERP vector (on the unit sphere)
    E_slerp = alpha * u1 + beta * u2  # shape (L, d)

    # 8. Fallback: Linear interpolation for zero-norm or near-parallel rows
    E_linear = (1.0 - t) * X1 + t * X2  # shape (L, d)
    fallback_mask = parallel_mask  # shape (L,1) => broadcastable

    # 9. Combine results
    #    E[i] = E_slerp[i] if not fallback_mask[i], else E_linear[i]
    # Note: torch.where requires matching shapes, so we expand the mask to (L,d).
    fallback_mask_full = fallback_mask.expand(-1, X1.shape[1])
    E = torch.where(fallback_mask_full, E_linear, E_slerp)

    return E, diverge_mask, norm1, norm2


def fake_minicache_merge(X1, X2, t=0.5, gamma=0.05):
    E, diverge_mask, n1, n2 = slerp_merge_rows_batch(X1, X2, t=t, gamma=gamma)
    diverge_mask = diverge_mask.squeeze(-1)
    E1 = torch.clone(E) * n1
    E1[~diverge_mask] = X1[~diverge_mask]
    E2 = torch.clone(E) * n2
    E2[~diverge_mask] = X2[~diverge_mask]
    return E1, E2


class FakeLayerMergingCache(DynamicCache):
    def __init__(
        self,
        merge_setup: xKVConfig,
        turboquant_registry=None,
        runtime_audit=None,
    ):
        """Simplified interface: num_heads and head_dim are inferred from the input tensors."""
        super().__init__()
        self.num_layers = merge_setup.num_layers
        self.merge_setup = merge_setup
        self.turboquant_registry = (
            {} if turboquant_registry is None else turboquant_registry
        )
        self.runtime_audit = runtime_audit
        
        # Configure the Quantizer
        if merge_setup.kv_quantizer == "rtn" and merge_setup.kv_bits < 16:
            self.configure_quantizer(
                kv_bits=merge_setup.kv_bits,
                group_size=merge_setup.group_size,
                sym=merge_setup.sym,
                clip_ratio=merge_setup.clip_ratio,
                hadamard=merge_setup.hadamard,
            )
        elif merge_setup.kv_quantizer == "turboquant_mse":
            self._validate_turboquant_codebooks()

        if hasattr(self, "layers"):
            cache_layout = "layers"
        elif hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            cache_layout = "key_cache/value_cache"
        else:
            cache_layout = "unknown"
        record_cache_created(self.runtime_audit, cache_layout)

    def _resolved_turboquant_codebook_path(self):
        codebook_path = self.merge_setup.turboquant_codebook_path
        if codebook_path is None:
            return None
        return str(Path(codebook_path).expanduser().resolve())

    def _validate_turboquant_codebooks(self):
        """Fail before prefill when a configured offline table is missing."""
        codebook_path = self._resolved_turboquant_codebook_path()
        required_codebooks = set()
        for group in self.merge_setup.layer_groups:
            if self.merge_setup.merge_key:
                required_codebooks.add(
                    (group.rank_k, self.merge_setup.quant_bits_for("k"))
                )
            if self.merge_setup.merge_value:
                required_codebooks.add(
                    (group.rank_v, self.merge_setup.quant_bits_for("v"))
                )
        for dimension, bits in sorted(required_codebooks):
            load_codebook(dimension, bits, codebook_path)

    def _get_turboquant_codec(self, rank, bits, device):
        """Return a shared immutable TurboQuant codec for one factor shape."""
        device = torch.device(device)
        codebook_path = self._resolved_turboquant_codebook_path()
        registry_key = (
            int(rank),
            int(bits),
            int(self.merge_setup.turboquant_rotation_seed),
            self.merge_setup.turboquant_chunk_size,
            codebook_path,
            str(device),
        )
        codec = self.turboquant_registry.get(registry_key)
        if codec is None:
            codec = TurboQuantMSE(
                dimension=rank,
                bits=bits,
                rotation_seed=self.merge_setup.turboquant_rotation_seed,
                chunk_size=self.merge_setup.turboquant_chunk_size,
                codebook_path=codebook_path,
                norm_dtype=torch.float32,
                compute_dtype=torch.float32,
            ).to(device=device)
            codec.eval()
            self.turboquant_registry[registry_key] = codec
        return codec

    def _quantizer_for(self, side, rank, device):
        if self.merge_setup.kv_quantizer == "turboquant_mse":
            bits = self.merge_setup.quant_bits_for(side)
            return self._get_turboquant_codec(rank, bits, device)
        return self.quantizer if hasattr(self, "quantizer") else None

    def get_layer_cache(self, layer_idx):
        """Read one layer from list-based or CacheLayer-based storage."""
        layers = getattr(self, "layers", None)
        if layers is not None:
            if layer_idx >= len(layers):
                raise RuntimeError(
                    "Cannot read an uninitialized Transformers cache layer"
                )
            cache_layer = layers[layer_idx]
            key = getattr(cache_layer, "keys", None)
            value = getattr(cache_layer, "values", None)
        elif hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            if layer_idx >= len(self.key_cache):
                raise RuntimeError(
                    "Cannot read an uninitialized Transformers cache layer"
                )
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
        else:
            raise RuntimeError(
                "Unsupported Transformers DynamicCache layout: expected "
                "key_cache/value_cache lists or a layers list"
            )
        if key is None or value is None:
            raise RuntimeError("Transformers cache layer has not been initialized")
        return key, value

    def _set_cached_kv(self, layer_idx, key, value):
        """Write one layer for both list-based and CacheLayer-based caches."""
        layers = getattr(self, "layers", None)
        if layers is not None:
            if layer_idx >= len(layers):
                raise RuntimeError(
                    "Cannot replace an uninitialized Transformers cache layer"
                )
            cache_layer = layers[layer_idx]
            if not hasattr(cache_layer, "keys") or not hasattr(
                cache_layer,
                "values",
            ):
                raise RuntimeError(
                    "Unsupported Transformers cache layer: missing keys/values"
                )
            cache_layer.keys = key
            cache_layer.values = value
            if hasattr(cache_layer, "is_initialized"):
                cache_layer.is_initialized = True
            return

        if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
            return

        raise RuntimeError(
            "Unsupported Transformers DynamicCache layout: expected "
            "key_cache/value_cache lists or a layers list"
        )

    def _should_merge(self, layer_idx):
        """Check if this layer is the last in its merge group using dictionary lookup."""
        group_info = self.merge_setup.get_group_for_layer(layer_idx)
        if group_info is not None: # No group found
            last_layer_idx_in_group = group_info.layers[-1]
            return layer_idx == last_layer_idx_in_group
        return False
        
    def is_value_merged(self):
        return self.merge_setup.merge_value

    def is_key_merged(self):
        return self.merge_setup.merge_key

    def update(self, key, value, layer_idx, mode='prefill', cos=None, sin=None, re_apply_rope=True):
        """Override update to hook into the fake SVD compression process."""
        super().update(key, value, layer_idx)

        if mode == 'decode':
            record_decode_update(self.runtime_audit)

        if mode == 'prefill':
            # Infer num_heads and head_dim from the key shape
            self.num_heads = key.shape[1]  # Shape: (batch_size, num_heads, seq_len, head_dim)
            self.head_dim = key.shape[3]

            # Apply grouped fake SVD if we have updated the last layer in the group
            if self._should_merge(layer_idx):
                self.grouped_layer_merging(layer_idx)
            
            group_info = self.merge_setup.get_group_for_layer(layer_idx)
            if group_info is not None: # grouped founded      
                if layer_idx == group_info.layers[-1]: # last layer in the group
                    for layer in group_info.layers:
                        pre_rope_key, layer_value = self.get_layer_cache(layer)
                        if re_apply_rope:
                            _, rotated_key = apply_rotary_pos_emb(
                                pre_rope_key,
                                pre_rope_key,
                                cos,
                                sin,
                            )
                            self._set_cached_kv(layer, rotated_key, layer_value)
            else: # no group found  
                pre_rope_key, layer_value = self.get_layer_cache(layer_idx)
                if re_apply_rope:
                    _, rotated_key = apply_rotary_pos_emb(
                        pre_rope_key,
                        pre_rope_key,
                        cos,
                        sin,
                    )
                    self._set_cached_kv(layer_idx, rotated_key, layer_value)
        return self.get_layer_cache(layer_idx)
    
    @torch.no_grad()
    def grouped_layer_merging(self, last_layer_idx):
        """Perform fake SVD on grouped layers, inferring dimensions from the tensors."""
        group_info = self.merge_setup.get_group_for_layer(last_layer_idx)
        if group_info is None:
            return  # No valid group found
        record_group_merge(self.runtime_audit)
        start_layer_idx, end_layer_idx = group_info.layers[0], group_info.layers[-1]       


        # Step 1: Collect keys and values for the layers in the group            
        keys, values = zip(*[
            self.get_layer_cache(i)
            for i in range(start_layer_idx, end_layer_idx + 1)
        ])
        split_sizes = [self.num_heads for _ in range(start_layer_idx, end_layer_idx + 1)]
        
        if self.merge_setup.layer_merge_impl == 'svd':
            # Step 2: Concatenate along the sequence length dimension
            combined_key = torch.cat(keys, dim=1)  # Shape: (batch_size, total_num_heads, seq_len, head_dim)
            combined_value = torch.cat(values, dim=1)

            # Step 3: Apply fake SVD (truncate and multiply back)
            #NOTE(brian1009): Experiment with fake SVD on key only for now
            if self.merge_setup.merge_key:
                key_quantizer = self._quantizer_for(
                    "k", group_info.rank_k, combined_key.device
                )
                combined_key = fake_svd(
                    combined_key.float(), 
                    rank=group_info.rank_k,
                    apply_hadamard_transform=getattr(self, 'use_hadamard', False),
                    quantizer=key_quantizer,
                    runtime_audit=self.runtime_audit,
                    audit_side="k",
                ).to(combined_key.dtype)
            if self.merge_setup.merge_value:
                value_quantizer = self._quantizer_for(
                    "v", group_info.rank_v, combined_value.device
                )
                combined_value = fake_svd(
                    combined_value.float(), 
                    rank=group_info.rank_v,
                    apply_hadamard_transform=getattr(self, 'use_hadamard', False),
                    quantizer=value_quantizer,
                    runtime_audit=self.runtime_audit,
                    audit_side="v",
                ).to(combined_value.dtype)

            # Step 4: Split and update the cache for each layer
            key_layers = torch.split(combined_key, split_sizes, dim=1)
            value_layers = torch.split(combined_value, split_sizes, dim=1)
        elif self.merge_setup.layer_merge_impl == 'slerp':
            assert len(keys) == 2 and len(values) == 2, "SLERP only supports group size 2"
            if self.merge_setup.merge_key: 
                key_reshaped = [k.reshape(-1, k.shape[-1]) for k in keys]
                keys_hat1, keys_hat2 = fake_minicache_merge(key_reshaped[0], key_reshaped[1], t=group_info.slerp_t, gamma=group_info.slerp_gamma)
                key_layers = (keys_hat1.reshape(keys[0].shape), keys_hat2.reshape(keys[1].shape))
            else:
                key_layers = keys
            
            if self.merge_setup.merge_value:
                value_reshaped = [v.reshape(-1, v.shape[-1]) for v in values]
                value_hat1, value_hat2 = fake_minicache_merge(value_reshaped[0], value_reshaped[1], t=group_info.slerp_t, gamma=group_info.slerp_gamma)
                value_layers = (value_hat1.reshape(values[0].shape), value_hat2.reshape(values[1].shape))
            else:
                value_layers = values

        else:
            raise NotImplementedError(f"Unknown implementation: {self.impl}")
        
        for idx, layer_idx in enumerate(range(start_layer_idx, end_layer_idx + 1)):
            self.update_cache(layer_idx, key_layers[idx], value_layers[idx])

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    def update_cache(self, layer_idx, key_approx, value_approx):
        """Update the cache with the approximated key and value tensors."""
        self._set_cached_kv(layer_idx, key_approx, value_approx)
    
    def get_max_length(self, layer_idx=None):
        """Return the cache limit across old and current Transformers APIs."""
        parent_get_max_length = getattr(super(), "get_max_length", None)
        if parent_get_max_length is None:
            # Transformers <= 4.57 uses None for an unbounded dynamic cache.
            # This method is also required by DeepSeek-Coder-V2-Lite-Instruct.
            return None

        # Current Transformers uses -1 for an empty/unbounded dynamic cache.
        if not self.layers:
            return -1
        return parent_get_max_length(layer_idx)

    def configure_quantizer(self, 
        kv_bits: int, 
        group_size: int, 
        sym: bool,
        clip_ratio: float,
        hadamard = False
    ):
        self.quantizer = Quantizer(kv_bits, group_size, sym, clip_ratio)
        self.use_hadamard = hadamard
