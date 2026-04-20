import itertools
import time

import numpy as np
import torch
import jax
import jax.numpy as jnp

from custom_models import ModelSpec


def jax_available():
    return jax is not None and jnp is not None


def require_jax():
    if not jax_available():
        raise SystemExit("JAX is required for the 'jax' method. Install jax and jaxlib.")


def _init_linear(key, in_dim, out_dim, scale=0.02):
    w_key, _ = jax.random.split(key)
    w = scale * jax.random.normal(w_key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def _linear(params, x):
    return x @ params["w"] + params["b"]


def _layer_norm(params, x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * params["g"] + params["b"]


def _init_layer_norm(key, dim):
    return {"g": jnp.ones((dim,)), "b": jnp.zeros((dim,))}


def _conv2d(params, x, stride=1):
    y = jax.lax.conv_general_dilated(
        x,
        params["w"],
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + params["b"]


def _max_pool(x, size=2, stride=2):
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, size, size, 1),
        window_strides=(1, stride, stride, 1),
        padding="SAME",
    )


def _init_mlp_params(key, input_shape, num_classes):
    c, h, w = input_shape
    in_features = c * h * w
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "fc1": _init_linear(k1, in_features, 256),
        "fc2": _init_linear(k2, 256, 128),
        "fc3": _init_linear(k3, 128, num_classes),
    }


def _apply_mlp(params, x):
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(_linear(params["fc1"], x))
    x = jax.nn.relu(_linear(params["fc2"], x))
    return _linear(params["fc3"], x)


def _init_cnn_params(key, in_channels, image_size, num_classes):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    pooled = image_size // 4
    return {
        "conv1": {"w": 0.02 * jax.random.normal(k1, (3, 3, in_channels, 32)), "b": jnp.zeros((32,))},
        "conv2": {"w": 0.02 * jax.random.normal(k2, (3, 3, 32, 64)), "b": jnp.zeros((64,))},
        "fc1": _init_linear(k3, 64 * pooled * pooled, 128),
        "fc2": _init_linear(k4, 128, num_classes),
    }


def _apply_cnn(params, x):
    x = jax.nn.relu(_conv2d(params["conv1"], x))
    x = _max_pool(x, 2, 2)
    x = jax.nn.relu(_conv2d(params["conv2"], x))
    x = _max_pool(x, 2, 2)
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(_linear(params["fc1"], x))
    return _linear(params["fc2"], x)


def _init_rnn_params(key, input_dim, hidden_dim, num_classes):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "rnn": {
            "wx": 0.02 * jax.random.normal(k1, (input_dim, hidden_dim)),
            "wh": 0.02 * jax.random.normal(k2, (hidden_dim, hidden_dim)),
            "b": jnp.zeros((hidden_dim,)),
        },
        "head": _init_linear(k3, hidden_dim, num_classes),
        "ln": _init_layer_norm(k4, hidden_dim),
    }


def _apply_rnn(params, x):
    wx = params["rnn"]["wx"]
    wh = params["rnn"]["wh"]
    b = params["rnn"]["b"]
    hidden_dim = b.shape[0]

    def step(h, x_t):
        h = jnp.tanh(x_t @ wx + h @ wh + b)
        return h, h

    h0 = jnp.zeros((x.shape[0], hidden_dim))
    _, h_seq = jax.lax.scan(step, h0, jnp.swapaxes(x, 0, 1))
    h_final = h_seq[-1]
    h_final = _layer_norm(params["ln"], h_final)
    return _linear(params["head"], h_final)


def _init_gru_params(key, input_dim, hidden_dim, num_classes):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "gru": {
            "wx": 0.02 * jax.random.normal(k1, (input_dim, 3 * hidden_dim)),
            "wh": 0.02 * jax.random.normal(k2, (hidden_dim, 3 * hidden_dim)),
            "b": jnp.zeros((3 * hidden_dim,)),
        },
        "head": _init_linear(k3, hidden_dim, num_classes),
    }


def _apply_gru(params, x):
    wx = params["gru"]["wx"]
    wh = params["gru"]["wh"]
    b = params["gru"]["b"]
    hidden_dim = b.shape[0] // 3

    def step(h, x_t):
        x_gates = x_t @ wx
        h_gates = h @ wh
        r_x, z_x, n_x = jnp.split(x_gates, 3, axis=-1)
        r_h, z_h, n_h = jnp.split(h_gates, 3, axis=-1)
        r_b, z_b, n_b = jnp.split(b, 3, axis=-1)
        r = jax.nn.sigmoid(r_x + r_h + r_b)
        z = jax.nn.sigmoid(z_x + z_h + z_b)
        n = jnp.tanh(n_x + r * n_h + n_b)
        h = (1.0 - z) * n + z * h
        return h, h

    h0 = jnp.zeros((x.shape[0], hidden_dim))
    _, h_seq = jax.lax.scan(step, h0, jnp.swapaxes(x, 0, 1))
    h_final = h_seq[-1]
    return _linear(params["head"], h_final)


def _init_lstm_params(key, input_dim, hidden_dim, num_classes):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "lstm": {
            "wx": 0.02 * jax.random.normal(k1, (input_dim, 4 * hidden_dim)),
            "wh": 0.02 * jax.random.normal(k2, (hidden_dim, 4 * hidden_dim)),
            "b": jnp.zeros((4 * hidden_dim,)),
        },
        "head": _init_linear(k3, hidden_dim, num_classes),
    }


def _apply_lstm(params, x):
    wx = params["lstm"]["wx"]
    wh = params["lstm"]["wh"]
    b = params["lstm"]["b"]
    hidden_dim = b.shape[0] // 4

    def step(state, x_t):
        h, c = state
        gates = x_t @ wx + h @ wh + b
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c = f * c + i * g
        h = o * jnp.tanh(c)
        return (h, c), h

    h0 = jnp.zeros((x.shape[0], hidden_dim))
    c0 = jnp.zeros((x.shape[0], hidden_dim))
    (_, _), h_seq = jax.lax.scan(step, (h0, c0), jnp.swapaxes(x, 0, 1))
    h_final = h_seq[-1]
    return _linear(params["head"], h_final)


def _init_mha_params(key, input_dim, embed_dim, num_heads, num_classes):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "q": _init_linear(k1, input_dim, embed_dim),
        "k": _init_linear(k2, input_dim, embed_dim),
        "v": _init_linear(k3, input_dim, embed_dim),
        "o": _init_linear(k4, embed_dim, embed_dim),
        "head": _init_linear(k5, embed_dim, num_classes),
    }


def _apply_mha(params, x, num_heads):
    embed_dim = params["o"]["w"].shape[0]
    head_dim = embed_dim // num_heads

    q = _linear(params["q"], x)
    k = _linear(params["k"], x)
    v = _linear(params["v"], x)

    def reshape_heads(y):
        b, t, _ = y.shape
        y = y.reshape((b, t, num_heads, head_dim))
        return jnp.transpose(y, (0, 2, 1, 3))

    qh = reshape_heads(q)
    kh = reshape_heads(k)
    vh = reshape_heads(v)

    scale = head_dim ** -0.5
    attn = jnp.einsum("bhqd,bhkd->bhqk", qh, kh) * scale
    attn = jax.nn.softmax(attn, axis=-1)
    ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, vh)
    ctx = jnp.transpose(ctx, (0, 2, 1, 3)).reshape((x.shape[0], x.shape[1], embed_dim))

    out = _linear(params["o"], ctx)
    pooled = jnp.mean(out, axis=1)
    return _linear(params["head"], pooled)


def _init_vit_params(
    key,
    image_size,
    patch_size,
    in_channels,
    embed_dim,
    depth,
    num_heads,
    num_classes,
):
    keys = jax.random.split(key, 4 + depth * 8)
    num_patches = (image_size // patch_size) ** 2
    params = {
        "patch": _init_linear(keys[0], patch_size * patch_size * in_channels, embed_dim),
        "cls": jax.random.normal(keys[1], (1, 1, embed_dim)) * 0.02,
        "pos": jax.random.normal(keys[2], (1, num_patches + 1, embed_dim)) * 0.02,
        "blocks": [],
        "head": _init_linear(keys[3], embed_dim, num_classes),
    }

    idx = 4
    for _ in range(depth):
        block = {
            "ln1": _init_layer_norm(keys[idx], embed_dim),
            "q": _init_linear(keys[idx + 1], embed_dim, embed_dim),
            "k": _init_linear(keys[idx + 2], embed_dim, embed_dim),
            "v": _init_linear(keys[idx + 3], embed_dim, embed_dim),
            "o": _init_linear(keys[idx + 4], embed_dim, embed_dim),
            "ln2": _init_layer_norm(keys[idx + 5], embed_dim),
            "mlp1": _init_linear(keys[idx + 6], embed_dim, embed_dim * 4),
            "mlp2": _init_linear(keys[idx + 7], embed_dim * 4, embed_dim),
        }
        params["blocks"].append(block)
        idx += 8
    return params


def _apply_vit(
    params,
    x,
    patch_size,
    num_heads,
    embed_dim,
):
    b, h, w, c = x.shape
    p = patch_size
    x = x.reshape(b, h // p, p, w // p, p, c)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = x.reshape(b, -1, p * p * c)

    x = _linear(params["patch"], x)
    cls = jnp.repeat(params["cls"], b, axis=0)
    x = jnp.concatenate([cls, x], axis=1)
    x = x + params["pos"][:, : x.shape[1], :]

    head_dim = embed_dim // num_heads

    for block in params["blocks"]:
        y = _layer_norm(block["ln1"], x)
        q = _linear(block["q"], y)
        k = _linear(block["k"], y)
        v = _linear(block["v"], y)

        def reshape_heads(y):
            y = y.reshape((b, y.shape[1], num_heads, head_dim))
            return jnp.transpose(y, (0, 2, 1, 3))

        qh = reshape_heads(q)
        kh = reshape_heads(k)
        vh = reshape_heads(v)

        scale = head_dim ** -0.5
        attn = jnp.einsum("bhqd,bhkd->bhqk", qh, kh) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, vh)
        ctx = jnp.transpose(ctx, (0, 2, 1, 3)).reshape((b, x.shape[1], embed_dim))

        attn_out = _linear(block["o"], ctx)
        x = x + attn_out

        y = _layer_norm(block["ln2"], x)
        y = jax.nn.gelu(_linear(block["mlp1"], y))
        y = _linear(block["mlp2"], y)
        x = x + y

    cls_out = x[:, 0, :]
    return _linear(params["head"], cls_out)


def _jax_model_factory(
    model_name,
    spec,
    key,
):
    
    """
    Args:
    - model_name: One of "mlp", "cnn", "rnn", "lstm", "gru", "mha", "vit"
    - spec: ModelSpec object with appropriate fields filled based on model type
    - key: JAX random key for parameter initialization
    """
    name = model_name.lower()

    if name == "mlp":
        params = _init_mlp_params(key, (spec.in_channels, spec.image_size, spec.image_size), spec.num_classes)
        return params, _apply_mlp, (spec.in_channels, spec.image_size, spec.image_size)
    if name == "cnn":
        params = _init_cnn_params(key, spec.in_channels, spec.image_size, spec.num_classes)
        return params, _apply_cnn, (spec.in_channels, spec.image_size, spec.image_size)
    if name == "rnn":
        params = _init_rnn_params(key, spec.seq_input_dim, spec.seq_hidden_dim, spec.num_classes)
        return params, _apply_rnn, (spec.seq_len, spec.seq_input_dim)
    if name == "lstm":
        params = _init_lstm_params(key, spec.seq_input_dim, spec.seq_hidden_dim, spec.num_classes)
        return params, _apply_lstm, (spec.seq_len, spec.seq_input_dim)
    if name == "gru":
        params = _init_gru_params(key, spec.seq_input_dim, spec.seq_hidden_dim, spec.num_classes)
        return params, _apply_gru, (spec.seq_len, spec.seq_input_dim)
    if name == "mha":
        num_heads = 4
        params = _init_mha_params(key, spec.seq_input_dim, spec.seq_hidden_dim, num_heads, spec.num_classes)
        apply_fn = lambda p, x: _apply_mha(p, x, num_heads=num_heads)
        return params, apply_fn, (spec.seq_len, spec.seq_input_dim)
    if name == "vit":
        patch_size = 4
        embed_dim = 192
        depth = 4
        num_heads = 6
        params = _init_vit_params(
            key,
            image_size=spec.image_size,
            patch_size=patch_size,
            in_channels=spec.in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=spec.num_classes,
        )
        apply_fn = lambda p, x: _apply_vit(
            p,
            x,
            patch_size=patch_size,
            num_heads=num_heads,
            embed_dim=embed_dim,
        )
        return params, apply_fn, (spec.in_channels, spec.image_size, spec.image_size)

    raise ValueError(f"Unsupported JAX model: {model_name}")


def _jax_prepare_batch(x, y, is_image):
    x_np = x.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(np.int32)
    if is_image:
        x_np = np.transpose(x_np, (0, 2, 3, 1))
    return jnp.asarray(x_np), jnp.asarray(y_np)


def _jax_dp_step(
    params,
    apply_fn,
    x,
    y,
    rng,
    max_grad_norm,
    noise_multiplier,
    lr,
):
    def loss_single(p, x_i, y_i):
        logits = apply_fn(p, x_i[None, ...])[0]
        loss = -jax.nn.log_softmax(logits)[y_i]
        return loss

    grad_fn = jax.vmap(jax.grad(loss_single), in_axes=(None, 0, 0))
    per_sample_grads = grad_fn(params, x, y)

    leaves, treedef = jax.tree_util.tree_flatten(per_sample_grads)
    sq_norms = None
    for g in leaves:
        axes = tuple(range(1, g.ndim))
        g_sq = jnp.sum(g ** 2, axis=axes)
        sq_norms = g_sq if sq_norms is None else sq_norms + g_sq
    norms = jnp.sqrt(sq_norms + 1e-12)
    clip_factors = jnp.minimum(1.0, max_grad_norm / norms)

    def clip_grad(g):
        reshape = (g.shape[0],) + (1,) * (g.ndim - 1)
        return g * clip_factors.reshape(reshape)

    clipped = jax.tree_util.tree_map(clip_grad, per_sample_grads)
    agg = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), clipped)

    noise_std = noise_multiplier * max_grad_norm / x.shape[0]
    flat_agg, treedef = jax.tree_util.tree_flatten(agg)
    keys = jax.random.split(rng, len(flat_agg) + 1)
    new_rng = keys[0]
    noisy = []
    for g, k in zip(flat_agg, keys[1:]):
        if noise_std > 0:
            g = g + noise_std * jax.random.normal(k, g.shape)
        noisy.append(g)
    noisy = jax.tree_util.tree_unflatten(treedef, noisy)

    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, noisy)

    logits = apply_fn(params, x)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean((preds == y).astype(jnp.float32))
    loss = jnp.mean(jax.vmap(loss_single, in_axes=(None, 0, 0))(params, x, y))

    return new_params, float(loss), float(acc), new_rng


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
):
    require_jax()
    key = jax.random.PRNGKey(0)
    params, apply_fn, _ = _jax_model_factory(model_name, spec, key)
    is_image = model_name in {"mlp", "cnn", "vit"}

    losses = []
    accs = []
    steps = 0

    stream = itertools.cycle(train_loader)
    for _ in range(warmup_steps):
        x, y = next(stream)
        x_jax, y_jax = _jax_prepare_batch(x, y, is_image)
        params, _, _, key = _jax_dp_step(
            params,
            apply_fn,
            x_jax,
            y_jax,
            key,
            max_grad_norm,
            noise_multiplier,
            lr,
        )

    start = time.perf_counter()
    for _ in range(max_steps):
        x, y = next(stream)
        x_jax, y_jax = _jax_prepare_batch(x, y, is_image)
        params, loss, acc, key = _jax_dp_step(
            params,
            apply_fn,
            x_jax,
            y_jax,
            key,
            max_grad_norm,
            noise_multiplier,
            lr,
        )
        losses.append(loss)
        accs.append(acc)
        steps += 1
        if steps >= max_steps:
            break

    elapsed = time.perf_counter() - start

    test_accs = []
    for x, y in test_loader:
        x_jax, y_jax = _jax_prepare_batch(x, y, is_image)
        logits = apply_fn(params, x_jax)
        preds = jnp.argmax(logits, axis=-1)
        test_accs.append(float(jnp.mean((preds == y_jax).astype(jnp.float32))))

    return {
        "train_loss": float(sum(losses) / max(len(losses), 1)),
        "train_acc": float(sum(accs) / max(len(accs), 1)),
        "test_acc": float(sum(test_accs) / max(len(test_accs), 1)),
        "seconds": elapsed,
        "steps": float(steps),
    }


__all__ = [
    "jax_available",
    "require_jax",
    "train_jax_dp",
]
