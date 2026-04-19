# Per-Sample Gradient Technique Comparison

## Methods

| Method | Idea | Strengths | Weaknesses | Compile Friendliness |
|---|---|---|---|---|
| Loop per sample | Compute grad one example at a time | Exact reference, easy to verify | Slow Python overhead | Low |
| Microbatch loop | Split batch into small chunks and loop | Lower memory vs full batch loop | Still Python-loop bound | Low |
| Hook-based (Opacus) | Register backward hooks to capture grad-sample tensors | Broad layer coverage, battle-tested | Hook overhead, harder to compile end-to-end | Medium-Low |
| `torch.func.vmap + grad` | Vectorize single-sample grad function over batch | High throughput, clean tensor graph | Needs vmap-compatible ops/layers | High |
| Johnson-Lindenstrauss projection | Project gradients to lower dimension for norm estimation | Can reduce memory/compute for clipping norms | Approximation error, added complexity | Medium |
| BackPACK | Custom extensions for per-sample statistics | Rich second-order stats ecosystem | Extra dependency and coverage limits | Medium |

## Practical guidance in this repo

- Use `diagnostic.py` for runtime and numerical comparisons between:
  - `loop`
  - `microbatch`
  - `vmap`
- Use `benchmark_layers.py` and `vit_benchmark.py` for end-to-end DP-SGD training throughput.
- If Opacus is installed, use it as a baseline for hook-based grad-sample collection.

## Recommendation

For this project's JIT objective, `vmap + grad` is the primary path because it produces a cleaner compile region than Python hooks and loop-based methods.
