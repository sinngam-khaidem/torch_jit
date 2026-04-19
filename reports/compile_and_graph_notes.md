# DP-SGD, `torch.compile`, `torch.func`, and Graph Modes

## Why `torch.func` helps

`torch.func.grad` + `torch.func.vmap` creates per-sample gradients without Python hooks. This usually forms a cleaner tensor program, which is easier for `torch.compile` to optimize.

## Static vs dynamic graph intuition

- Static-style regions (predictable control flow and tensor shapes) compile best.
- Dynamic Python-side logic (hooks, shape-dependent branching, object mutation) introduces graph breaks.

## PyTorch and XLA context

- JAX pipelines typically rely on XLA to compile a fused training step.
- In PyTorch, `torch.compile` provides analogous acceleration opportunities, but actual speedups depend on model structure, backend, and warmup costs.
- This repo reports both total and steady-state timings for fair interpretation.

## Practical benchmark interpretation

- `total_seconds`: includes compile/warmup overhead.
- `steady_seconds` or throughput columns: better for long training runs.
- Expect compile path to look slower for very short runs and faster in steady state.
