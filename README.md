# JIT Accelerated DP-SGD in PyTorch

VMAP + `torch.compile` DP-SGD pipeline.

- We implemented JIT and VMAP based DPSGD Update that integrates with most of the existing components/modules within PyTorch.
- Custom layers/models of MLP, CNN, RNN, GRU, LSTM for JIT compatibility.
- Contains scripts for accuracy and privacy accounting.

**Note: JIT Compilation for RNN, GRU, LSTM only works on CUDA.**
**All evaluation results are in results.ipynb**

The main class `CustomPrivacyEngine` is defined in `torch_privacy.py` & you can check our evaluation results within `results.ipynb`
## Running guide

```bash
python evaluate.py --datasets mnist,cifar10 --max-steps 10
python benchmark_layers.py --max-steps 10 --warmup-steps 3
```


## References
https://docs.pytorch.org/docs/stable/func.html

https://docs.pytorch.org/docs/stable/generated/torch.vmap.html

https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html

https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/