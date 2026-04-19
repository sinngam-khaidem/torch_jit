# DPSGD Implementation Comparison Notes

## Compared systems

- Non-private SGD (baseline)
- Custom DP-SGD with `torch.func.vmap + grad`
- Custom DP-SGD with `torch.compile`
- Opacus DP-SGD (optional, if installed)
- JAX / Custom-TF references (external baselines)

## Core design differences

| System | Per-sample gradients | Clipping | Noise | Compiler story |
|---|---|---|---|---|
| Non-private SGD | N/A | N/A | N/A | Standard graph |
| Custom PyTorch DP-SGD | `vmap(grad(single_loss))` | Flat or layerwise | Gaussian | `torch.compile` over tensor-heavy path |
| Opacus | Backward hooks | Layer-wise/flat in engine | Gaussian | Hooks can reduce compileability |
| JAX DP-SGD | `vmap` + XLA | Tensorized clipping | Gaussian | XLA whole-program compilation |
| Custom-TF DP-SGD | Usually vectorized grads + XLA/Graph | Tensorized clipping | Gaussian | Strong graph mode path |

## Current repo coverage

- Implemented: `MLP`, `CNN`, `RNN`, `LSTM`, `GRU`, `MHA`, `TinyViT`
- Implemented: clipping variants (`flat`, `layerwise`)
- Implemented: Poisson subsampling option
- Implemented: timing + accuracy + epsilon reporting scripts

## How to run comparisons

- Full benchmark across datasets/models:
  - `python evaluate.py --datasets mnist,cifar10,fakedata --max-steps 20`
- New layer benchmark:
  - `python benchmark_layers.py --max-steps 25 --warmup-steps 3`
- ViT speed benchmark:
  - `python vit_benchmark.py --dataset cifar10 --max-steps 20 --warmup-steps 3`
- Clipping/Poisson exploration:
  - `python alternatives_dp.py --dataset mnist --model cnn --max-steps 20`
- Per-sample gradient diagnostics:
  - `python diagnostic.py --model lstm --batch-size 32`
