import itertools
import time

import numpy as np
import torch

import haiku as hk
import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.tree_util import tree_flatten, tree_unflatten


from custom_models import ModelSpec


def jax_available():
    return True


def require_jax():
    if not jax_available():
        raise SystemExit("JAX is required for the 'jax' method. Install jax and jaxlib.")


def _haiku_mlp(features, num_classes):
    x = hk.Flatten()(features)
    x = hk.Linear(256)(x)
    x = jax.nn.relu(x)
    x = hk.Linear(128)(x)
    x = jax.nn.relu(x)
    return hk.Linear(num_classes)(x)


def _haiku_cnn(features, num_classes):
    return hk.Sequential(
        [
            hk.Conv2D(16, (8, 8), padding="SAME", stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding="VALID"),
            hk.Conv2D(32, (4, 4), padding="VALID", stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding="VALID"),
            hk.Flatten(),
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(num_classes),
        ]
    )(features)


def _haiku_rnn(features, hidden_dim, num_classes):
    input_proj = hk.Linear(hidden_dim)
    hidden_proj = hk.Linear(hidden_dim, with_bias=False)

    def step(h, x_t):
        h = jnp.tanh(input_proj(x_t) + hidden_proj(h))
        return h, h

    h0 = jnp.zeros((features.shape[0], hidden_dim))
    _, h_seq = jax.lax.scan(step, h0, jnp.swapaxes(features, 0, 1))
    h_final = h_seq[-1]
    return hk.Linear(num_classes)(h_final)


def _haiku_gru(features, hidden_dim, num_classes):
    core = hk.GRU(hidden_dim)
    h0 = core.initial_state(features.shape[0])
    outs, _ = hk.dynamic_unroll(core, features, h0, time_major=False)
    h_final = outs[:, -1, :]
    return hk.Linear(num_classes)(h_final)


def _haiku_lstm(features, hidden_dim, num_classes):
    core = hk.LSTM(hidden_dim)
    h0 = core.initial_state(features.shape[0])
    outs, _ = hk.dynamic_unroll(core, features, h0, time_major=False)
    h_final = outs[:, -1, :]
    return hk.Linear(num_classes)(h_final)



def _build_haiku_model(model_name, spec):
    name = model_name.lower()
    if name == "mlp":
        return hk.transform(lambda x: _haiku_mlp(x, spec.num_classes))
    if name == "cnn":
        return hk.transform(lambda x: _haiku_cnn(x, spec.num_classes))
    if name == "rnn":
        return hk.transform(lambda x: _haiku_rnn(x, spec.seq_hidden_dim, spec.num_classes))
    if name == "gru":
        return hk.transform(lambda x: _haiku_gru(x, spec.seq_hidden_dim, spec.num_classes))
    if name == "lstm":
        return hk.transform(lambda x: _haiku_lstm(x, spec.seq_hidden_dim, spec.num_classes))
    
    raise ValueError(f"Unsupported JAX model: {model_name}")


def _dummy_input_for_spec(spec):
    name = spec.model_name.lower()
    if name in {"mlp", "cnn", "vit"}:
        return jnp.zeros((1, spec.image_size, spec.image_size, spec.in_channels), dtype=jnp.float32)
    return jnp.zeros((1, spec.seq_len, spec.seq_input_dim), dtype=jnp.float32)


def _multiclass_loss(apply_fn, params, batch):
    inputs, targets = batch
    logits = apply_fn(params, inputs)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))


def _clipped_grad(loss_fn, params, l2_norm_clip, single_example_batch):
    inputs, targets = single_example_batch
    if inputs.ndim in (2, 3):
        inputs = inputs[None, ...]
        targets = jnp.reshape(targets, (1,))
    grads = grad(lambda p: loss_fn(p, (inputs, targets)))(params)
    nonempty_grads, tree_def = tree_flatten(grads)
    per_param = jnp.stack([jnp.linalg.norm(g.ravel()) for g in nonempty_grads])
    total_norm = jnp.linalg.norm(per_param)
    divisor = jnp.maximum(total_norm / l2_norm_clip, 1.0)
    normalized = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized)


def _private_grad_vmap(loss_fn, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    clipped = vmap(lambda eg: _clipped_grad(loss_fn, params, l2_norm_clip, eg))(batch)
    clipped_flat, treedef = tree_flatten(clipped)
    aggregated = [g.sum(0) for g in clipped_flat]
    rngs = random.split(rng, len(aggregated))
    noised = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated)
    ]
    normalized = [g / batch_size for g in noised]
    return tree_unflatten(treedef, normalized)


def _private_grad_no_vmap(loss_fn, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    grads = [_clipped_grad(loss_fn, params, l2_norm_clip, eg) for eg in batch]
    grads_flat = [tree_flatten(g)[0] for g in grads]
    treedef = tree_flatten(grads[0])[1]
    stacked = [jnp.stack([g[i] for g in grads_flat]) for i in range(len(grads_flat[0]))]
    aggregated = [g.sum(0) for g in stacked]
    rngs = random.split(rng, len(aggregated))
    noised = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated)
    ]
    normalized = [g / batch_size for g in noised]
    return tree_unflatten(treedef, normalized)


def _dp_sgd_step_jax_vmap(params, batch, rng, lr, max_grad_norm, noise_multiplier, batch_size, loss_fn):
    grads = _private_grad_vmap(
        loss_fn, params, batch, rng, max_grad_norm, noise_multiplier, batch_size
    )
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def _dp_sgd_step_jax_no_vmap(params, batch, rng, lr, max_grad_norm, noise_multiplier, batch_size, loss_fn):
    grads = _private_grad_no_vmap(
        loss_fn, params, batch, rng, max_grad_norm, noise_multiplier, batch_size
    )
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def _prepare_fast_batch(x, y, is_image):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    elif hasattr(y, "numpy"):
        y = y.numpy()
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)
    if is_image:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {x.shape}")
        if x.shape[-1] not in (1, 3) and x.shape[1] in (1, 3):
            x = np.transpose(x, (0, 2, 3, 1))
    return x, y


def _jax_model_factory(model_name, spec, key):
    require_jax()
    model = _build_haiku_model(model_name, spec)
    dummy = _dummy_input_for_spec(spec)
    params = model.init(key, dummy)

    def apply_fn(p, x):
        return model.apply(p, None, x)

    return params, apply_fn, dummy.shape[1:]


def train_jax_dp(
    model_name,
    spec,
    train_loader,
    test_loader,
    max_steps,
    warmup_steps,
    lr,
    max_grad_norm,
    noise_multiplier,
    use_vmap=True,
    use_jit=True,
):
    require_jax()
    key = random.PRNGKey(0)
    params, apply_fn, _ = _jax_model_factory(model_name, spec, key)
    is_image = model_name in {"mlp", "cnn", "vit"}

    def loss_fn(p, batch):
        return _multiclass_loss(apply_fn, p, batch)

    def step_fn(params, batch, rng):
        grads = (_private_grad_vmap if use_vmap else _private_grad_no_vmap)(
            loss_fn, params, batch, rng, max_grad_norm, noise_multiplier, batch[0].shape[0]
        )
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

    if use_jit:
        step_fn = jit(step_fn)

    stream = itertools.cycle(train_loader)
    for _ in range(warmup_steps):
        x, y = next(stream)
        x_np, y_np = _prepare_fast_batch(x, y, is_image)
        x_jax = jax.device_put(x_np)
        y_jax = jax.device_put(y_np)
        key = random.fold_in(key, 1)
        params = step_fn(params, (x_jax, y_jax), key)

    batch_x = []
    batch_y = []
    for _ in range(max_steps):
        x, y = next(stream)
        x_np, y_np = _prepare_fast_batch(x, y, is_image)
        batch_x.append(x_np)
        batch_y.append(y_np)

    x_steps = jax.device_put(np.stack(batch_x, axis=0))
    y_steps = jax.device_put(np.stack(batch_y, axis=0))
    rngs = random.split(key, max_steps)

    def scan_step(carry, inputs):
        params = carry
        x_step, y_step, rng = inputs
        params = step_fn(params, (x_step, y_step), rng)
        logits = apply_fn(params, x_step)
        loss = _multiclass_loss(apply_fn, params, (x_step, y_step))
        acc = jnp.mean((jnp.argmax(logits, axis=-1) == y_step).astype(jnp.float32))
        return params, (loss, acc)

    def run_scan(params, xs, ys, rngs):
        return jax.lax.scan(scan_step, params, (xs, ys, rngs))

    if use_jit:
        run_scan = jit(run_scan)

    start = time.perf_counter()
    params, (losses, accs) = run_scan(params, x_steps, y_steps, rngs)
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - start

    test_accs = []
    for x, y in test_loader:
        x_np, y_np = _prepare_fast_batch(x, y, is_image)
        x_jax = jax.device_put(x_np)
        y_jax = jax.device_put(y_np)
        logits = apply_fn(params, x_jax)
        preds = jnp.argmax(logits, axis=-1)
        test_accs.append(float(jnp.mean((preds == y_jax).astype(jnp.float32))))

    return {
        "train_loss": float(jnp.mean(losses)) if max_steps else 0.0,
        "train_acc": float(jnp.mean(accs)) if max_steps else 0.0,
        "test_acc": float(sum(test_accs) / max(len(test_accs), 1)),
        "seconds": elapsed,
        "steps": float(max_steps),
    }
