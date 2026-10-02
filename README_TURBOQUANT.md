# xKV + TurboQuant

This fork of [xKV](https://github.com/abdelfattah-lab/xKV) (Apache-2.0; the original README is [README.md](README.md)) adds **4-bit quantization of xKV's low-rank KV factors**, a **physically nibble-packed cache**, and a **fused Triton dequantization kernel**.

Qwen2.5-14B-Instruct-1M, 32K context, batch size 1, one RTX 6000 Ada:

- The KV cache is **13.4× smaller** than the dense BF16 cache (5.96 → 0.44 GiB).
- Quantizing the factors to 4 bits costs **0.04 pp** of RULER accuracy (10-task macro, 0.9232 → 0.9228).
- The fused kernel dequantizes **about 10× faster** than the PyTorch reference and is bit-identical to it. Together with a per-group dequantization cache it takes packed decode from 135 to 66 ms/token, the same speed as keeping the factors in BF16.

Caveats, detailed in [Scope and limitations](#scope-and-limitations): at batch size 1 the model weights dominate GPU memory, so total memory drops by 16%, and decode is about 1.17× slower than dense (66 vs 56.5 ms/token).

## How it works

1. xKV factorizes the keys (and values) of each group of 4 consecutive layers as `X ≈ A·B` with an SVD (rank 384 for K, 576 for V): one large left factor `A` of shape `[tokens, rank]` per group and a small right factor `B` per layer.
2. [TurboQuant-MSE](https://arxiv.org/abs/2504.19874) quantizes the rows of `A` to 4 bits after a deterministic random rotation, using offline Lloyd-Max codebooks (no calibration data). The inverse rotation is folded into `B`, so decoding needs no extra rotation. Two codes share one byte, and each row keeps an FP32 norm.
3. `PackedXKVCache` keeps only the packed codes and `B` resident. When a layer group is needed, one Triton kernel (`dequant_fused`) unpacks and dequantizes `A` for the whole group. The 4 layers of the group reuse it, and only the most recently used group stays dequantized.
4. Decode attention runs on the compressed prefix with xKV's Triton kernel (extended to return the log-sum-exp) and on a dense tail of recent tokens with FlashAttention-2. The two partial results are merged through their log-sum-exps.

## What is added

| Path | Content |
|---|---|
| `xKV/customized_cache/turboquant.py` | TurboQuant-MSE codec, offline codebooks (`assets/`), and physical 4-bit packing (`pack_turboquant_codes`, `unpack_turboquant_codes`) |
| `evaluate/eval_acc.py`, `xKV/`, `xKV_SR/` | `--kv_quantizer turboquant_mse --k_bits 4 --v_bits 4` for xKV and xKV-SR accuracy runs (simulated quantization) |
| `scripts/factorized_xkv_cache.py` | `BF16FactorXKVCache` and `PackedXKVCache`: persistent factors plus a dense decode tail |
| `scripts/xkv_wired_cache.py` | `XKVWiredCache`: dense decode tail for a compressed prefix kept by the caller |
| `efficiency/ops/fused_attention/rope/v1_dequant_fused.py` | Triton kernel: 4-bit unpack, codebook lookup and norm in one launch |
| `efficiency/ops/fused_attention/rope/v1.py` | Decode attention that also returns the log-sum-exp |
| `scripts/eval_ruler_*.py`, `measure_memory_timeline.py`, `bench_*.py` | The experiments below |

## Results

Setup: Qwen2.5-14B-Instruct-1M, xKV-4 (group size 4, `rank_k=384`, `rank_v=576`), TurboQuant K4/V4, batch size 1, one RTX 6000 Ada (torch 2.6.0, Triton 3.2.0, transformers 4.51.0, flash-attn 2.7.4.post1).

### KV-cache memory

32,496-token RULER prompt, measured with `torch.cuda.memory_allocated()` minus the weights-only baseline (27.51 GiB), by `scripts/measure_memory_timeline.py`:

| | Dense (BF16 KV) | xKV, BF16 factors | xKV + TurboQuant, packed |
|---|---:|---:|---:|
| KV cache after compression, before decode | 5.96 GiB | 0.91 GiB (6.6×) | **0.44 GiB (13.4×)** |
| KV cache after 11 decode steps | 5.96 GiB | not measured | 0.50 GiB (11.8×) |

The packed cache grows by about 0.06 GiB once decoding starts, because one layer group's dequantized `A` stays resident. The 13.4× figure is the state right after compression; 11.8× is the steady state during decode.

### Total GPU memory and decode speed

`scripts/eval_ruler_factorized_cache.py`, `niah_multivalue`, 6 samples of about 32K tokens. The first sample is left out of the decode speed because it includes one-time Triton JIT warm-up (119.6 ms/token for packed, against 66 for the next five):

| | Dense (FlashAttention-2) | xKV, BF16 factors | xKV + TurboQuant, packed |
|---|---:|---:|---:|
| Total GPU memory, ready to decode | 33.38 GiB | 28.41 GiB (−14.9%) | 27.96 GiB (−16.3%) |
| Decode, ms/token | 56.5 | 66.0 (1.17×) | 66.1 (1.17×) |
| Score (6 samples) | 1.00 | 0.96 | 0.96 |

Packing the factors to 4 bits costs no decode speed compared with keeping them in BF16. Both xKV variants miss one of the four needles in the sixth sample (0.75) that dense answers correctly; the BF16 factors miss it too, so it is not caused by the quantization. SVD, quantization and packing add about 1.2 s once per request, on top of a prefill of about 15.7 s.

### Decode optimizations

`scripts/bench_packed_decode_ablation.py` runs the same decode on the same compressed cache with each optimization switched on in turn, in one process and in rotating order. The generated tokens are identical. 5 samples, median ms/token with the range:

| `PackedXKVCache` dequantization | ms/token | Saved |
|---|---:|---:|
| PyTorch, on every layer call (original) | 134.7 (132.6–135.2) | |
| PyTorch, once per layer group (single-entry cache) | 82.3 (81.2–82.6) | 52.4 ms (39%) |
| Fused Triton kernel, once per layer group (shipped) | 66.1 (65.3–74.7) | 16.2 ms (20%) |

One of the five fused samples ran slower (74.7); the other four are within 65.3–66.3.

### Fused dequantization kernel

`scripts/bench_dequant_fused.py` (no model needed; random BF16 factors encoded with the real codec; 4 runs, `kv_len` 32,496 to 32,768):

| | PyTorch reference | Fused kernel | Speedup |
|---|---:|---:|---:|
| K side, rank 384 | 0.66–0.73 ms | 0.066–0.072 ms | 9.6–11.1× |
| V side, rank 576 | 0.87–0.90 ms | 0.074–0.089 ms | 9.9–11.8× |

The output matches the reference exactly (max abs diff 0).

### RULER accuracy

`scripts/eval_ruler_wired_cache.py`: 32K context, 20 samples per task, 10 tasks (listed below), real kernel path:

| | xKV (BF16 factors) | xKV + TurboQuant K4/V4 |
|---|---:|---:|
| Macro average | 0.9232 | 0.9228 |

<details>
<summary>Per-task scores</summary>

| Task | xKV | xKV + K4/V4 |
|---|---:|---:|
| niah_single_1 | 1.0000 | 1.0000 |
| niah_single_2 | 0.9500 | 1.0000 |
| niah_multikey_1 | 1.0000 | 1.0000 |
| niah_multikey_2 | 1.0000 | 1.0000 |
| niah_multiquery | 0.9875 | 0.9750 |
| niah_multivalue | 0.9375 | 0.9125 |
| vt | 0.9400 | 0.9400 |
| fwe | 0.9167 | 0.9000 |
| qa_1 | 0.7500 | 0.8000 |
| qa_2 | 0.7500 | 0.7000 |

</details>

Quantization in the accuracy runs is simulated (the factors are quantized and immediately dequantized to BF16). The packed cache stores the same codes.

## Reproduce

Setup follows [README.md](README.md). Generate the RULER data once; `create_dataset.sh` now also builds the 32K length:

```bash
bash scripts/build_datasets.sh
# build_datasets.sh skips a model whose folder already exists. If evaluate/data/ruler/data/qwen
# comes from an earlier 64K-only build, build the 32K data directly:
(cd evaluate/data/ruler && bash create_dataset.sh Qwen/Qwen2.5-14B-Instruct-1M qwen)
```

Kernel (no model, needs a CUDA GPU):

```bash
python scripts/bench_dequant_fused.py
```

Decode speed and GPU memory of the three variants (use at least 2 samples: the first is warm-up and is left out of the speed):

```bash
M=Qwen/Qwen2.5-14B-Instruct-1M
python scripts/eval_ruler_factorized_cache.py --model $M --task niah_multivalue --num-samples 6 --dense
python scripts/eval_ruler_factorized_cache.py --model $M --task niah_multivalue --num-samples 6
python scripts/eval_ruler_factorized_cache.py --model $M --task niah_multivalue --num-samples 6 --packed
```

Effect of each decode optimization:

```bash
python scripts/bench_packed_decode_ablation.py --model Qwen/Qwen2.5-14B-Instruct-1M --num-samples 5
```

KV-cache memory ratio (each mode in its own process):

```bash
for mode in dense bf16factor packed; do
  python scripts/measure_memory_timeline.py --model Qwen/Qwen2.5-14B-Instruct-1M --mode $mode
done
python - <<'EOF'
import json

GiB = 1024 ** 3

def allocated_at(timeline, label):
    t = next(t for t, name in timeline["markers"] if name == label)
    return next(a for ts, a, _ in timeline["samples"] if ts >= t) / GiB

def kv_gib(mode, ready_marker):
    tl = json.load(open(f"results/memory_timeline_{mode}.json"))
    return allocated_at(tl, ready_marker) - allocated_at(tl, "model_loaded")  # minus weights

dense = kv_gib("dense", "ready_to_decode")
packed = kv_gib("packed", "dense_cache_freed_ready_to_decode")
print(f"dense KV {dense:.3f} GiB | packed KV {packed:.3f} GiB | {dense / packed:.1f}x smaller")
EOF
```

Accuracy (10 tasks; drop `--quantize` for the xKV baseline):

```bash
for t in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multiquery \
         niah_multivalue vt fwe qa_1 qa_2; do
  python scripts/eval_ruler_wired_cache.py --model Qwen/Qwen2.5-14B-Instruct-1M \
      --task $t --num-samples 20 --quantize | grep avg_score
done
```

The upstream harness also runs TurboQuant, for xKV and xKV-SR (simulated quantization):

```bash
python evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M \
    --dataset_name "ruler/niah_single_1" --datalen 32768 --flash2 \
    --method xkv --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 \
    --start_layer_idx 0 --end_layer_idx -1 \
    --kv_quantizer turboquant_mse --k_bits 4 --v_bits 4
```

CPU unit tests:

```bash
pytest tests/test_turboquant_packing.py tests/test_turboquant_mse.py tests/test_turboquant_codebook.py \
       tests/test_turboquant_xkv_integration.py tests/test_factorized_xkv_cache.py
```

## Scope and limitations

- **The 13.4× is for the KV cache only.** At batch size 1 the 27.5 GiB of weights dominate, so total GPU memory drops by 16%, not 13×.
- **Decode is slower than dense**: 66 ms/token against 56.5 ms/token (1.17×) at batch size 1.
- **Batch size 1, greedy decoding, 4-bit only.** Anything else, and exceeding the 512-token dense tail, raises an error instead of continuing silently.
- **Only the left factors `A` are quantized**; the right factors `B` stay BF16.
- **Accuracy**: 20 samples per task, so one sample is 5 pp on a task, and the 0.04 pp difference is far below the noise. The figure measures only the cost of quantizing the factors. xKV's own compression costs accuracy against full attention: an earlier run of the same 10 tasks with 32 samples each scored 0.9469 for full attention and 0.9099 for xKV (not reproduced by these scripts).
- Tested only with Qwen2.5-14B-Instruct-1M at 32K context, and with the numbers above taken on an RTX 6000 Ada.

## Credits

xKV is by its authors ([paper](https://arxiv.org/abs/2503.18893), [README.md](README.md)); TurboQuant is described in "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874). This fork is released under the same Apache-2.0 license ([LICENSE](LICENSE)).
