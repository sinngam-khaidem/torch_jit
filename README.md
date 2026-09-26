# JIT-Accelerated DP-SGD in PyTorch

Differentially private SGD is usually slow. The per-sample gradient clipping that makes it
private also breaks the batching that makes training fast, so you end up paying a "privacy tax"
of anywhere from 2x to 20x in wall-clock time.

This project removes most of that tax in PyTorch by combining `torch.func.vmap` (for per-sample
gradients) with `torch.compile` (to fuse the whole DP update into one kernel region).

It is a PyTorch port of [Subramani et al. (2021)](https://arxiv.org/abs/2010.09063), who showed
the same idea works in JAX and TensorFlow. We rebuild it on PyTorch 2.x with TorchInductor as
the compiler backend.

## Working.

`CustomPrivacyEngine` has two modes:

| Mode | What it does | Where it runs |
|---|---|---|
| `custom_eager` | Per-sample gradients via `vmap(grad(f))`, no compilation | CUDA, CPU, MPS |
| `custom_compiled` | vmap + clipping + noise + parameter update, all inside one `torch.compile` region | CUDA (required for RNN / GRU / LSTM) |


```python
from torch_privacy import CustomPrivacyEngine, DPConfig
from custom_models import ModelSpec, build_model

model = build_model(ModelSpec("cnn", num_classes=10, in_channels=1, image_size=28)).to("cuda")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

engine = CustomPrivacyEngine(model, config=DPConfig(
    max_grad_norm=1.0,
    noise_multiplier=1.1,
    vmap_chunk_size=32,
))

for x, y in loader:
    stats = engine.dp_step(x.cuda(), y.cuda(), optimizer, use_compile=True)
```

Clipping can be `flat` (one global norm per sample, the standard DP-SGD recipe) or `layerwise`.

### Custom Model Reimplementation.

`custom_models.py` has its own MLP, CNN, RNN, GRU and LSTM. That is deliberate. PyTorch's
`nn.RNN` / `nn.LSTM` / `nn.GRU` dispatch to opaque cuDNN kernels that `torch.func` transforms
cannot see through, so they are re-implemented as plain functional loops. Those loops are
happy to be vmapped, and TorchInductor can fuse them.


## Results

Benchmarked on a CUDA GPU, 1000 steps at batch size 128, `lr=1e-3`, `max_grad_norm=1.0`,
`noise_multiplier=1.1`, `vmap_chunk_size=32`. MLP and CNN train on MNIST; the LSTM trains on a
synthetic sequence task (length 64, input dim 64, hidden dim 128).

| Model | Backend | Train Loss | Train Acc | Test Acc | Seconds |
|---|---|---:|---:|---:|---:|
| MLP | non-private | 2.2862 | 0.1512 | 0.2309 | 7.12 |
| MLP | **ours (compiled)** | 2.2948 | 0.1375 | 0.1500 | **6.70** |
| MLP | ours (eager) | 2.2992 | 0.1121 | 0.1253 | 21.07 |
| MLP | JAX | 2.2938 | 0.1283 | 0.1711 | 4.00 |
| MLP | Opacus | 2.2989 | 0.1252 | 0.1264 | 15.35 |
| CNN | non-private | 2.2714 | 0.2769 | 0.5015 | 7.98 |
| CNN | **ours (compiled)** | 2.2958 | 0.1031 | 0.1373 | **10.76** |
| CNN | ours (eager) | 2.2935 | 0.1271 | 0.1474 | 35.02 |
| CNN | JAX | 2.3123 | 0.0371 | 0.0571 | 5.34 |
| CNN | Opacus | 2.2969 | 0.1145 | 0.1761 | 18.83 |
| LSTM | non-private | 2.3027 | 0.1009 | 0.1113 | 57.32 |
| LSTM | **ours (compiled)** | 2.3034 | 0.1079 | 0.1025 | **19.16** |
| LSTM | ours (eager) | 2.3041 | 0.1029 | 0.1064 | 323.13 |
| LSTM | JAX | 2.3154 | 0.0903 | 0.0967 | 19.72 |
| LSTM | Opacus | 2.3019 | 0.1046 | 0.1143 | 94.68 |


- **Compilation is worth 3x to 17x.** Eager to compiled: 3.1x on MLP, 3.3x on CNN, 16.9x on LSTM.
- **The privacy tax mostly disappears.** Compiled DP-SGD matches non-private training on the MLP
  (6.70s vs 7.12s) and costs only 1.35x on the CNN.
- **Beats Opacus everywhere.** 2.3x on MLP, 1.8x on CNN, 4.9x on LSTM.
- **Matches JAX on LSTM** (19.16s vs 19.72s) and stay within 1.7x to 2.0x of it on the
  smaller image models.

These are timing runs, not convergence runs. 1000 steps at `lr=1e-3`
is not enough for any of these models to learn much, which is why the accuracies are low and
close to each other. The point of the table is the last column.

Full run, including the compile diagnostics (`torch._dynamo.explain` reports 1 graph, 0 graph
breaks, 623 ops for the LSTM step) is in `notebooks/results.ipynb`.

## Getting started

```bash
pip install -r requirements.txt
```

Then run a benchmark:

```bash
# all models, all backends
python evaluate_all.py --datasets mnist --batch-size 128 --max-steps 200

# image models only
python evaluate_image_models.py --datasets mnist,cifar10

# sequence models only (RNN / GRU / LSTM)
python evaluate_seq_models.py --seq-len 32 --hidden-dim 128
```

Or open `notebooks/results.ipynb` to walk through the comparison interactively.

## Layout

```
torch_privacy.py          CustomPrivacyEngine, DPConfig, clipping + noise
custom_models.py          JIT-friendly MLP, CNN, RNN, GRU, LSTM
opacus_models.py          same architectures, for the Opacus baseline
jax_models.py             same architectures in Haiku, for the JAX baseline
privacy_accounting.py     epsilon accounting (RDP)
evaluate_*.py             benchmark drivers
notebooks/                results and exploratory work
```

## References

- Subramani, Vadivelu & Kamath. [Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization](https://arxiv.org/abs/2010.09063). NeurIPS 2021.
- [`torch.func` documentation](https://docs.pytorch.org/docs/stable/func.html)
- [Per-sample gradients tutorial](https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html)
- [Optimizing CUDA RNNs with TorchScript](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/)
