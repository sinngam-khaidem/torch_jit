from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def flat_aggregate(per_sample_grad_list, max_grad_norm, noise_multiplier, batch_size, device, eps=1e-12):
    """
    Clip each sample using a single global norm across all layers.

    Args:
    - per_sample_grad_list: List of per-sample gradients for each parameter.
    - max_grad_norm: The maximum allowed norm for the gradients.
    - noise_multiplier: The standard deviation of the Gaussian noise to be added, relative to max_grad_norm.
    - batch_size: The number of samples in the batch (used to scale the noise).
    - eps: A small constant to prevent division by zero when computing norms.
    """
    # Compute per-sample norms across all parameters in a vectorized way
    sq_norms = torch.stack([g.flatten(1).pow(2).sum(dim=1) for g in per_sample_grad_list]).sum(0)
    norms = torch.sqrt(sq_norms + eps)

    # Clip factors per sample
    clip_factor = torch.clamp(max_grad_norm / norms, max=1.0)

    # Apply clipping and aggregate
    clipped = []
    for g in per_sample_grad_list:
        reshape = (g.size(0),) + (1,) * (g.dim() - 1)
        factor = clip_factor.view(reshape)
        clipped.append((g * factor).mean(dim=0))

    noise_std = noise_multiplier * max_grad_norm / batch_size

    device_type = device.type if isinstance(device, torch.device) else str(device)
    # MPS backend does not support foreach APIs, so we need to check before using them.
    use_foreach = device_type != "mps" and hasattr(torch, "_foreach_add")

    if noise_std > 0:
        if use_foreach:
            noise = [torch.randn_like(g, device=device) * noise_std for g in clipped]
            noisy = torch._foreach_add(clipped, noise)
        else:
            noisy = [g + torch.randn_like(g, device=device) * noise_std for g in clipped]
    else:
        noisy = clipped

    return tuple(noisy)


def layerwise_aggregate(per_sample_grad_list, max_grad_norm, noise_multiplier, batch_size, device, eps=1e-12):
    """
    Clips each sample independently per layer.
    """
    clipped = []

    for g in per_sample_grad_list:
        norms = torch.sqrt(g.flatten(1).pow(2).sum(dim=1) + eps)
        factors = torch.clamp(max_grad_norm / (norms + eps), max=1.0)
        reshape = (g.size(0),) + (1,) * (g.dim() - 1)
        clipped.append((g * factors.view(reshape)).mean(dim=0))

    noise_std = noise_multiplier * max_grad_norm / max(batch_size, 1)
    device_type = device.type if isinstance(device, torch.device) else str(device)
    use_foreach = device_type != "mps" and hasattr(torch, "_foreach_add")
    if noise_std > 0:
        if use_foreach:
            noise = [torch.randn_like(g, device=device) * noise_std for g in clipped]
            noisy = torch._foreach_add(clipped, noise)
        else:
            noisy = [g + torch.randn_like(g, device=device) * noise_std for g in clipped]
    else:
        noisy = clipped

    return tuple(noisy)


@dataclass
class DPConfig:
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    vmap_chunk_size: Optional[int] = 32
    force_compile_mps: bool = True


class CustomPrivacyEngine:
    def __init__(self, model, config: Optional[DPConfig] = None):
        self.model = model
        self.config = config or DPConfig()
        self._compiled_aggregators = {}
        self._compiled_steps = {}
        self._compiled_step_failed = set()
        self.param_names = [name for name, _ in self.model.named_parameters()]

    # Compute per-sample gradients efficiently using function transforms.
    # https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html
    def compute_single_loss(self, params, buffers, x, y, loss_fn):
        """
        Compute the loss of the model given a single sample (x, y) rather than a batch of samples.

        Args:
            - params: A dictionary of the model's parameters.
            - buffers: A dictionary of the model's buffers.
            - x: A single input sample (shape [C, H, W] for an image).
            - y: The corresponding label for the input sample (shape []).
        Returns:
            - The loss value for the given sample.
        """
        # Adds a batch dimension to x and y so they can be passed through the model.
        x = x.unsqueeze(0) # [C, H, W] -> [1, C, H, W]
        y = y.unsqueeze(0) # [] -> [1]

        # functional_call to treat an `nn.Module` like a function.
        logits = functional_call(self.model, (params, buffers), (x,))
        return loss_fn(logits, y)

    def make_per_sample_grad_fn(self, loss_fn):
        def single_loss(params, buffers, x, y):
            return self.compute_single_loss(params, buffers, x, y, loss_fn)

        # `grad` transform is used to create a new function `grad_fn` that computes the gradients w.r.t the first argument of `compute_loss`.
        # `grad_fn` will compute gradient for a single (x, y) pair. We use `vmap` to compute gradients over an entire batch of (x, y) pairs in parallel.
        grad_fn = grad(single_loss)
        return vmap(grad_fn, in_dims=(None, None, 0, 0), chunk_size=self.config.vmap_chunk_size)

    def make_per_sample_gradients(self, x, y, loss_fn=None):
        """
        Compute per-sample gradients for a batch of inputs (x, y) efficiently using vmap.

        Args:
            - x: A batch of input samples (shape [B, C, H, W] for images).
            - y: The corresponding labels for the input samples (shape [B]).
            - chunk_size: Optional integer to specify the chunk size for vmap to control memory usage.
        Returns:
            - A tuple containing:
                - params: A dictionary of the model's parameters.
                - buffers: A dictionary of the model's buffers.
                - per_sample_grads: A dictionary where each key corresponds to a parameter name and the value is a tensor of shape [B, ...] containing the per-sample gradients for that parameter.
        """

        loss_fn = loss_fn or F.cross_entropy
        params = {k: v for k, v in self.model.named_parameters()}
        buffers = {k: v for k, v in self.model.named_buffers()}

        per_sample_fn = self.make_per_sample_grad_fn(loss_fn)

        per_sample_grads = per_sample_fn(params, buffers, x, y)
        return params, buffers, per_sample_grads
    
    
    def get_aggregator(self, clip_mode: str):
        if clip_mode == "flat":
            fn = flat_aggregate
        elif clip_mode == "layerwise":
            fn = layerwise_aggregate
        else:
            raise ValueError(f"Unsupported clip_mode: {clip_mode}")
        
        return fn

    def get_compiled_step(self, clip_mode: str, loss_fn):
        """
        Creates a JIT compiled version of the DP-SGD step function.
        """
        # To avoid recompilation overhead on subsequent calls.
        cache_key = f"{clip_mode}_compiled_step"
        if cache_key in self._compiled_steps:
            return self._compiled_steps[cache_key]


        aggregator = self.get_aggregator(clip_mode)
        per_sample_fn = self.make_per_sample_grad_fn(loss_fn)

        def dp_sgd_step(params, buffers, batch, max_grad_norm, noise_multiplier, lr):
            x, y = batch

            per_sample = per_sample_fn(params, buffers, x, y)
            grads = [per_sample[name] for name in self.param_names]
            
            noisy = aggregator(grads, max_grad_norm, noise_multiplier, x.shape[0], x.device)

            # SGD update using foreach for lower overhead (skip on mps)
            params_list = [params[name] for name in self.param_names]

            device_type = x.device.type
            use_foreach = device_type != "mps" and hasattr(torch, "_foreach_add")

            if use_foreach:
                new_params_list = torch._foreach_add(params_list, list(noisy), alpha=-lr)
            else:
                new_params_list = [p - lr * g for p, g in zip(params_list, noisy)]

            return dict(zip(self.param_names, new_params_list))

        compiled = torch.compile(
            dp_sgd_step,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )
        self._compiled_steps[cache_key] = compiled
        return compiled

    def clip_and_aggregate(self, per_sample_grads, clip_mode: str = "flat"):
        """
        Clips and aggregates per-sample gradients.

        Args:
            - per_sample_grads: A dictionary of per-sample gradients for each parameter.
            - clip_mode: "flat" or "layerwise" clipping strategy.
            - use_compile: Whether to use torch.compile for the aggregation step.
        Returns:
            - A dictionary of aggregated (and noised) gradients for each parameter, ready to be applied to the model.
        """
        grads = [per_sample_grads[name] for name in self.param_names]
        batch_size = grads[0].shape[0]
        device = grads[0].device

        aggregator = self.get_aggregator(clip_mode=clip_mode)
        
        noisy = aggregator(
            grads,
            self.config.max_grad_norm,
            self.config.noise_multiplier,
            batch_size,
            device,
        )
        return {name: g for name, g in zip(self.param_names, noisy)}


    def dp_step(
        self,
        x,
        y,
        optimizer,
        loss_fn=None,
        clip_mode: str = "flat",
        use_compile: bool = False,
    ):
        """
        Args:
            - x: Batch of input samples.
            - y: Batch of target labels.
            - optimizer: Optimizer to apply the DP update.
            - loss_fn: Loss function to compute per-sample losses (default: cross-entropy).
            - clip_mode: "flat" or "layerwise" clipping strategy.
            - use_compile: Whether to use torch.compile for the aggregation step.
        Returns:
            A dictionary containing the batch loss, accuracy, and batch size.
        
        """
        loss_fn = loss_fn or F.cross_entropy

        used_compiled = False

        # Attempt to use the compiled DP step if enabled and not previously failed for this clip mode.
        if use_compile and clip_mode not in self._compiled_step_failed:
            try:
                params = {k: v.detach() for k, v in self.model.named_parameters()}
                buffers = {k: v for k, v in self.model.named_buffers()}

                lr = float(optimizer.param_groups[0].get("lr", 0.0))

                compiled_step = self.get_compiled_step(clip_mode=clip_mode, loss_fn=loss_fn)

                updated = compiled_step(
                    params,
                    buffers,
                    (x, y),
                    float(self.config.max_grad_norm),
                    float(self.config.noise_multiplier),
                    lr,
                )
                
                for name, param in self.model.named_parameters():
                    # print(f"Updating param: {name}, requires_grad={param.requires_grad}")
                    if param.requires_grad:
                        param.data.copy_(updated[name])

                used_compiled = True
                #print(f"Used compiled DP step for clip_mode={clip_mode}")
            except Exception as e:
                print(f"Compiled DP step failed for clip_mode={clip_mode}, falling back to eager. Exception: {e}")
                print(e.print_stack())
                self._compiled_step_failed.add(clip_mode)
                used_compiled = False

        # If compilation failed or is disabled, use the eager path.
        if not used_compiled:
            optimizer.zero_grad(set_to_none=True)

            _, _, per_sample = self.make_per_sample_gradients(x, y, loss_fn=loss_fn)
            noisy_grads = self.clip_and_aggregate(per_sample, clip_mode=clip_mode)

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.grad = noisy_grads[name]

            optimizer.step()
            #print(f"Used eager DP step for clip_mode={clip_mode}")
        
        with torch.no_grad():
            logits = self.model(x)
            loss = loss_fn(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": float(loss.item()),
            "acc": float(acc.item()),
            "batch_size": float(y.numel()),
        }
    


__all__ = [
    "DPConfig",
    "CustomPrivacyEngine",
]


if __name__ == "__main__":
    import argparse
    from layers import ModelSpec, build_model

    parser = argparse.ArgumentParser(description="Test CustomPrivacyEngine.")
    
    parser.add_argument(
        "--models",
        type=str,
        default="mlp,cnn,rnn,lstm,gru,mha,vit",
        help="Comma-separated model names to test.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Compile full DP-SGD step when possible.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    
    for model_name in model_names:
        if model_name in {"rnn", "lstm", "gru", "mha"}:
            spec = ModelSpec(model_name=model_name, seq_input_dim=64, seq_len=32, seq_hidden_dim=128, num_classes=10)
        elif model_name == "vit":
            spec = ModelSpec(model_name="vit", in_channels=3, image_size=32, num_classes=10)
        elif model_name == "mlp":
            spec = ModelSpec(model_name="mlp", in_channels=1, image_size=28, num_classes=10)
        else:
            spec = ModelSpec(model_name="cnn", in_channels=1, image_size=28, num_classes=10)

        model = build_model(spec).to(device)

        cfg = DPConfig(max_grad_norm=1.0, noise_multiplier=1.1, vmap_chunk_size=32)
        engine = CustomPrivacyEngine(model, cfg)

        
        if model_name in {"rnn", "lstm", "gru", "mha"}:
            x = torch.randn(args.batch_size, spec.seq_len, spec.seq_input_dim, device=device)
        elif model_name == "vit":
            x = torch.randn(args.batch_size, 3, 32, 32, device=device)
        else:
            x = torch.randn(args.batch_size, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = None

        for _ in range(args.steps):
            stats = engine.dp_step(x, y, optimizer, use_compile=args.compile)
        print(f"{model_name}: {stats}")
