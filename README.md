# JIT Accelerated DP-SGD in PyTorch

This repository contains a VMAP + `torch.compile` DP-SGD pipeline with:

- VMAP/JIT-friendly layers for `RNN`, `LSTM`, `GRU`, and `MultiHeadAttention`
- Tiny ViT benchmarking for the "final boss" speed comparison
- Scripts for runtime/accuracy/privacy accounting comparisons
- Diagnostic scripts for per-sample gradient techniques
- Alternative experiments for clipping modes and Poisson subsampling

## Main files

- `layers.py`: model and layer definitions (`MLP`, `CNN`, recurrent layers, attention, `TinyViT`)
- `torch_privacy.py`: functional DP-SGD engine (`vmap + grad`, clipping, noise, optional compile)
- `privacy_accounting.py`: epsilon accounting utilities
- `evaluate.py`: multi-dataset benchmark (MNIST/CIFAR10/FakeData)
- `benchmark_layers.py`: benchmark newly implemented recurrent/attention layers
- `vit_benchmark.py`: ViT-focused DP-SGD speed benchmark
- `alternatives_dp.py`: clipping and Poisson subsampling exploration
- `diagnostic.py`: per-sample gradient method comparison

## Quick start

```bash
python evaluate.py --datasets mnist,cifar10,fakedata --max-steps 10
python benchmark_layers.py --max-steps 10 --warmup-steps 3
python vit_benchmark.py --dataset fakedata --max-steps 10 --warmup-steps 3
python alternatives_dp.py --dataset mnist --model cnn --max-steps 10
python diagnostic.py --model lstm --batch-size 16
```

## Notes

- Opacus comparisons are optional and only run when `opacus` is installed.
- CIFAR10 automatically falls back to FakeData if download is unavailable.




- cnn 
- mlp
- rnn
- lstm 
- mha 
- gru -> eager
- vit -> eager