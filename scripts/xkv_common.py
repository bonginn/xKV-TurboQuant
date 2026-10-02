"""Helpers shared by the evaluation and measurement scripts."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "efficiency"))
from svd.svd_api import SVDConfig, run_svd  # noqa: E402


def load_model_and_tokenizer_fa2(model_path):
    """Load a causal LM in BF16 on CUDA with FlashAttention-2, plus its tokenizer.

    FlashAttention-2 (not sdpa) is the attention backend the repo README
    recommends for Qwen models, and it is the dense baseline used by every
    comparison in these scripts.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", attn_implementation="flash_attention_2",
    )
    model.eval()
    return model, tokenizer


def balanced_gauge_svd(combined_bf16, rank, method="full"):
    """Rank-`rank` balanced-gauge factorization X ~= A @ B with A = U sqrt(S)
    and B = sqrt(S) V^H.

    combined_bf16: [batch, num_heads, seq_len, head_dim] -- the K (or V) of
    one layer group, concatenated along the head axis. Returns BF16
    A [batch, seq_len, rank] and B [batch, rank, num_heads * head_dim].

    method:
      "full"   -- torch.linalg.svd (exact, but slow at long context)
      "cholqr" -- randomized SVD with Cholesky-QR (efficiency/svd); a drop-in
                  replacement that returns U/S/Vh in the same shape
                  convention, already truncated to `rank`
    """
    bs, nh, sl, hd = combined_bf16.shape
    reshaped = combined_bf16.transpose(1, 2).reshape(bs, sl, nh * hd)
    if method == "full":
        U, S, Vh = torch.linalg.svd(reshaped.float(), full_matrices=False)
        U_trunc, S_trunc, Vt_trunc = U[:, :, :rank], S[:, :rank], Vh[:, :rank, :]
    elif method == "cholqr":
        U_trunc, S_trunc, Vt_trunc = run_svd(reshaped, SVDConfig(method="cholqr", rank=rank))
        U_trunc, S_trunc, Vt_trunc = U_trunc.float(), S_trunc.float(), Vt_trunc.float()
    else:
        raise ValueError(f"unknown method {method!r}")
    sqrt_S = torch.sqrt(S_trunc)
    A = (U_trunc * sqrt_S.unsqueeze(1)).to(torch.bfloat16).contiguous()
    B = (sqrt_S.unsqueeze(-1) * Vt_trunc).to(torch.bfloat16)
    return A, B
