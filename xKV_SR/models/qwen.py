################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


import torch
import torch.nn.functional as F
import gc
import time

import transformers
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
transformers.logging.set_verbosity_error()

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single, sample_token
from .prompt_template import Templates, Chat_Templates
from .base import LLM
from .merge_configs import generate_consecutive_palu_config

class Qwen2Layer:
    def __init__(self, layer_idx) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.bq :torch.Tensor = None
        self.bk :torch.Tensor = None
        self.bv :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: Qwen2DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        # bias for qkv
        self.bq = hf_layer.self_attn.q_proj.bias.detach()
        self.bk = hf_layer.self_attn.k_proj.bias.detach()
        self.bv = hf_layer.self_attn.v_proj.bias.detach()

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

        self.bq = self.bq.to(device, non_blocking=True)
        self.bk = self.bk.to(device, non_blocking=True)
        self.bv = self.bv.to(device, non_blocking=True)

class Qwen2(LLM):
    def __init__(self, 
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        batch_size :int = 1,
        max_length :int = 64*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        # minference
        minference=False,
        # ShadowKV
        sparse_budget: int = 2048,
        rank: int = 160,
        chunk_size: int = 8,
        # xKV
        start_layer_idx: int = 0,
        end_layer_idx: int = -1,
        group_size: int = 1,
        rank_k: int = 512,
        rank_v: int = 768,
        fake_svd: bool = False,
    ) -> None:
        
        assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = Qwen2Config.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        # RoPE frequencies are read from the loaded Hugging Face rotary
        # embedding in init_parameters().  Do not require the legacy
        # Qwen2Config.rope_theta attribute, which is absent in some
        # Transformers versions.

        self.init_parameters()
        self.attn_mode = attn_mode
        self.minference = minference

        self.merge_config = generate_consecutive_palu_config(
            start_layer=start_layer_idx,
            # generate_consecutive_palu_config uses an inclusive end index.
            end_layer=end_layer_idx if end_layer_idx != -1 else self.num_layers - 1,
            group_size=group_size,
            rank_k=rank_k,
            rank_v=rank_v
        )

        self.ctx_template = Templates['qwen']
        self.chat_template = Chat_Templates['qwen']

        self.fake_svd = fake_svd
        self.init_kv_cache(sparse_budget, chunk_size, self.config, rank=rank, merge_config=self.merge_config)

    def _set_cos_sin_cache(self, hf_model):
        """Build a version-compatible cache through Hugging Face RoPE."""
        # Transformers >= 4.46 stores one shared rotary embedding on the
        # model. Older versions stored it on each attention layer.
        rotary_emb = getattr(hf_model.model, "rotary_emb", None)
        uses_model_level_api = rotary_emb is not None
        if not uses_model_level_api:
            first_attention = hf_model.model.layers[0].self_attn
            rotary_emb = getattr(first_attention, "rotary_emb", None)

        if rotary_emb is None:
            raise RuntimeError(
                "Unable to locate the Qwen2 rotary embedding on either "
                "`model.rotary_emb` or "
                "`model.layers[0].self_attn.rotary_emb`."
            )

        # Keep the module on its existing device. Some dynamic-RoPE versions
        # retain a non-buffer alias of the original frequencies, so moving the
        # module can leave its internal tensors split across devices.
        inv_freq = getattr(rotary_emb, "inv_freq", None)
        rotary_device = inv_freq.device if inv_freq is not None else torch.device("cpu")

        # Calling Hugging Face's implementation preserves attention scaling,
        # YaRN, and dynamic RoPE updates that a manual inv_freq path can miss.
        with torch.inference_mode():
            if uses_model_level_api:
                dummy_x = torch.empty(
                    (1, 1, self.hidden_size),
                    device=rotary_device,
                    dtype=self.dtype,
                )
                position_ids = torch.arange(
                    self.max_length,
                    device=rotary_device,
                    dtype=torch.long,
                ).unsqueeze(0)
                cos_cache, sin_cache = rotary_emb(dummy_x, position_ids)
            else:
                dummy_x = torch.empty(
                    (1, self.num_key_value_heads, 1, self.head_dim),
                    device=rotary_device,
                    dtype=self.dtype,
                )
                cos_cache, sin_cache = rotary_emb(
                    dummy_x,
                    seq_len=self.max_length,
                )

        # Newer model-level RoPE returns [1, length, head_dim], whereas the
        # legacy attention-level implementation returns [length, head_dim].
        if cos_cache.ndim == 3 and cos_cache.shape[0] == 1:
            cos_cache = cos_cache[0]
        if sin_cache.ndim == 3 and sin_cache.shape[0] == 1:
            sin_cache = sin_cache[0]

        expected_shape = (self.max_length, self.head_dim)
        if tuple(cos_cache.shape) != expected_shape or tuple(sin_cache.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected Qwen2 RoPE cache shapes: "
                f"cos={tuple(cos_cache.shape)}, sin={tuple(sin_cache.shape)}, "
                f"expected={expected_shape}."
            )

        return (
            cos_cache.to(device=self.device, dtype=self.dtype).contiguous(),
            sin_cache.to(device=self.device, dtype=self.dtype).contiguous(),
        )

    def init_parameters(self):
        hf_model = Qwen2ForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache(hf_model)
        self.layers :list[Qwen2Layer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen2Layer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: Qwen2Layer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, buffer.wq, bias=buffer.bq)
        key_states = F.linear(hidden_states, buffer.wk, bias=buffer.bk)
        value_states = F.linear(hidden_states, buffer.wv, bias=buffer.bv)
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        return query_states, key_states, value_states
    
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: Qwen2Layer
    ):  
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = F.linear(hidden_states, buffer.up_proj)
        gate = F.silu(F.linear(hidden_states, buffer.gate_proj))
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states
    
    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_single(x, self.cos_cache, self.sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb(q, k, self.cos_cache, self.sin_cache, position_ids)
