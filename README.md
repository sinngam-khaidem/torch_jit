# JIT Accelerated DP-SGD in PyTorch

VMAP + `torch.compile` DP-SGD pipeline with:

- We implemented JIT and VMAP friendly layer and models of MLP, CNN, RNN, GRU, LSTM, MultiheadAttention and ViT.
- Contains scripts for accuracy and privacy accounting.
- Diagnostic scripts for per-sample gradient techniques & JIT compilation using torch._dynamo.explain.

![Evaluation of MLP, CNN and VIT on MNIST and CIFAR10](torch-jit/Resources/report_100_steps.png)

## Main files

- `layers.py`: model and layer definitions.
- `torch_privacy.py`: functional DP-SGD engine (`vmap + grad`, clipping, noise, optional compile)
- `privacy_accounting.py`: epsilon accounting utilities
- `evaluate.py`: multi-dataset benchmark (MNIST/CIFAR10)
- `benchmark_seq_layers.py`: Benchmarking script for sequential/recurrent layers.
- `vit_benchmark.py`: Benchmarking script for TinyViT.
- `diagnostic.py`: per-sample gradient method & jit compilation.

## Running guide

```bash
python evaluate.py --datasets mnist,cifar10 --max-steps 10
python benchmark_layers.py --max-steps 10 --warmup-steps 3
python vit_benchmark.py --dataset fakedata --max-steps 10 --warmup-steps 3
```