# TurboQuant — HuggingFace-Compatible KV Cache Quantization

A from-scratch PyTorch implementation of **TurboQuant** (Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026), which combines **PolarQuant**-style rotation + Lloyd-Max scalar quantization with an optional **1-bit QJL residual correction**, wired directly into HuggingFace `transformers`' `Cache` API as a drop-in replacement for `DynamicCache`.

## Acknowledgments
This implementation draws heavily on [0xSero/turboquant](https://github.com/0xsero/turboquant), including some code used directly with minimal or no modification. Credit to 0xSero for the original implementation work — this repo should be read as a derivative/study built on top of theirs, not an independent implementation.

## What it does

Standard KV cache quantization pays a memory tax for storing per-block normalization constants (zero points, scales) in full precision. This implementation avoids that by:

1. **Rotating** each key/value vector with a random sign-flip (Rademacher vector) followed by a Fast Walsh-Hadamard Transform — an O(d log d) stand-in for a dense random rotation matrix, which spreads the vector's energy evenly across coordinates so no per-block calibration is needed.
2. **Scalar-quantizing** each rotated coordinate against a precomputed Lloyd-Max codebook, solved numerically against the known Beta-distribution shape that rotated unit vectors follow at dimension `d`.
3. **Optionally correcting the residual** (`TurboQuantResidual`) by spending one more bit on a QJL sign-sketch of the leftover error after MSE quantization, then adding the reconstructed correction back at dequant time.

The result plugs directly into `model.generate(..., past_key_values=TurboQuantCache(...))` with no other changes to the model or generation loop.

## Repo structure

| File | Contents |
|---|---|
| `codebook.py` | Lloyd-Max codebook solver for the rotated-coordinate distribution; computes centroids/boundaries via numerical integration + fixed-point iteration, caches results to disk as JSON per `(d, bits)` pair |
| `rotations.py` | `generateRademacher`, `FWHT` (Fast Walsh-Hadamard Transform), `forward_rotation`/`backward_rotation` (Rademacher sign-flip + FWHT), `generateQJLMatrix` (dense Gaussian `S` for the QJL sketch); also includes an unused QR-based `generate_rotation_matrix` |
| `TurboQuantOperations.py` | Core quantizer classes: `NoOpQuantizer`, `TurboQuantMSE` (rotation + Lloyd-Max codebook, MSE-optimal), `TurboQuantResidual` (MSE quantizer at `bits-1` + 1-bit QJL residual correction, "prod"-style); bit-packing/unpacking utilities; `TurboMSEPack`/`TurboQJLPack` storage tuples |
| `TQCache.py` | `TurboQuantCache` (subclasses `transformers.cache_utils.Cache`) and `TurboQuantLayer` (subclasses `CacheLayerMixin`) — the HF-compatible cache object, with independent quantizer choices for keys and values |
| `test.py` | End-to-end inference driver against `Qwen/Qwen2.5-3B-Instruct`, plus (commented out) an earlier standalone bias/reconstruction test harness used during development |
| `__init__.py` | Empty — package marker |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from TurboQ.TQCache import TurboQuantCache
from TurboQ.TurboQuantOperations import TurboQuantMSE, TurboQuantResidual

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct").to("cuda")

cache = TurboQuantCache(
    device=model.device,
    dim=128,                        # head_dim
    num_key_heads=model.config.num_key_value_heads,
    key_quantizer=TurboQuantMSE,        # or TurboQuantResidual
    value_quantizer=TurboQuantMSE,
    config=model.config,
    kquant_size=3,                  # bits per key coordinate
    vquant_size=3,                  # bits per value coordinate
)

encoded = tokenizer("...", return_tensors="pt").to("cuda")
output = model.generate(
    **encoded,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=500,
)
```

`key_quantizer` / `value_quantizer` are left `None` by default, in which case `TQCache` falls back to `NoOpQuantizer` (full-precision passthrough) — useful for A/B testing against an unquantized baseline.

## Design notes

- Codebooks are computed once per `(dim, bits)` pair and cached to `codebooks/codebook_d{d}_b{bits}.json`, so repeated runs at the same configuration skip the Lloyd-Max solve.
- `TurboQuantResidual` internally instantiates a `TurboQuantMSE` at `bits - 1` and computes the residual (`x - dequant(mse(x))`) itself — callers pass the raw key/value vector, not a pre-computed residual.
- The QJL sketch matrix `S` is `d × d` (one sign measurement per residual dimension), matching the ratio used in the original QJL/TurboQuant papers — this is intentional, not a sizing bug, since QJL is designed to give a low-variance *unbiased inner-product* estimator, not a low-error standalone vector reconstruction.

## Known limitation

Empirical testing against Qwen2.5-3B-Instruct found that `TurboQuantResidual`'s QJL correction **increases** end-to-end reconstruction error relative to plain MSE-only quantization (`TurboQuantMSE`) at low bit-widths (b ≤ 3-4), rather than improving it. This matches independent findings reported elsewhere in the community for the same bit-width regime and small head dimensions, attributed to variance from the 1-bit correction outweighing its unbiasedness benefit once passed through softmax. **Recommendation: use `TurboQuantMSE` for both keys and values at low bit budgets; treat `TurboQuantResidual` as experimental.**

## Requirements

`torch`, `transformers` (recent version exposing `CacheLayerMixin` / `DynamicSlidingWindowLayer` in `cache_utils`), `numpy`, `scipy`, `matplotlib` (optional, for diagnostic plots in `test.py`).

## References

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. arXiv:2504.19874
- Han, Kacham, Karbasi, Mirrokni, Zandieh. *PolarQuant: Quantizing KV Caches with Polar Transformation.* AISTATS 2026. arXiv:2502.02617
- Zandieh, Daliri, Han. *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* AAAI 2025. arXiv:2406.03482
