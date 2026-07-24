# JIT-Accelerated DP-SGD in PyTorch

A high-performance implementation of Differentially Private Stochastic Gradient Descent (DP-SGD)
for PyTorch, combining `torch.func.vmap` with `torch.compile` to eliminate the traditional
privacy tax on training speed.

This project is a PyTorch extension of Subramani et al. (2021), who demonstrated that
vectorization and JIT compilation can mostly eliminate DP-SGD runtime overhead in JAX and
TensorFlow. We port their methodology to the modern PyTorch 2.x ecosystem using TorchInductor
as the compiler backend.

## Overview

The core of this project is the `CustomPrivacyEngine` class (`torch_privacy.py`), which offers
two operational modes:

- **Eager mode** (`custom_eager`): Per-sample gradients via `vmap(grad(f))` without compilation.
  Portable across CUDA, CPU, and MPS devices.
- **Compiled mode** (`custom_compiled`): The entire DP-SGD update step — including vmap,
  clipping, noise injection, and parameter update — is wrapped in a single `torch.compile`
  region for maximum performance. **Requires CUDA for RNN, GRU, and LSTM architectures.**

Custom JIT-compatible implementations are provided for MLP, CNN, RNN, GRU, and LSTM. Standard
PyTorch `nn.RNN`/`nn.LSTM`/`nn.GRU` modules use opaque cuDNN backends that are incompatible
with `torch.func` transforms, so these have been reimplemented as pure functional loops amenable
to both `vmap` and kernel fusion by TorchInductor.

Evaluation results and timing comparisons across backends are available in `results.ipynb`.

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

## References

- Subramani, Vadivelu & Kamath. [Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization.](https://arxiv.org/abs/2010.09063) NeurIPS 2021.
- PyTorch `torch.func` documentation: https://docs.pytorch.org/docs/stable/func.html
- PyTorch per-sample gradients tutorial: https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html
- Optimizing CUDA RNNs with TorchScript: https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/s̄