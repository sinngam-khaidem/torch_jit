# JIT Accelerated DP-SGD in PyTorch

VMAP + `torch.compile` DP-SGD pipeline.

- We implemented JIT and VMAP based DPSGD Update that integrates with most of the existing components/modules within PyTorch.
- Custom layers/models of MLP, CNN, RNN, GRU, LSTM, MultiheadAttention and ViT for JIT compatibility.
- Contains scripts for accuracy and privacy accounting.

![Evaluation of MLP, CNN and VIT on MNIST and CIFAR10](/torch-jit/Resources/report_100_steps.png)

**Note: JIT Compilation for RNN, GRU, LSTM, MultiheadAttention and ViT only works on CUDA.**

## Main files

- `layers.py`: model and layer definitions.
- `torch_privacy.py`: functional DP-SGD engine (`vmap + grad`, clipping, noise, optional compile)
- `privacy_accounting.py`: epsilon accounting utilities
- `evaluate.py`: multi-dataset benchmark (MNIST/CIFAR10)
- `benchmark_seq_layers.py`: Benchmarking script for sequential/recurrent layers.
- `vit_benchmark.py`: Benchmarking script for TinyViT.

## Running guide

```bash
python evaluate.py --datasets mnist,cifar10 --max-steps 10
python benchmark_layers.py --max-steps 10 --warmup-steps 3
python vit_benchmark.py --dataset fakedata --max-steps 10 --warmup-steps 3
```


## References
https://docs.pytorch.org/docs/stable/func.html

https://docs.pytorch.org/docs/stable/generated/torch.vmap.html

https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html

# https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/