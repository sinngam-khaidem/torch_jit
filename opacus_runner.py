import itertools
import time
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def train_opacus(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int,
    warmup_steps: int,
    max_grad_norm: float,
    noise_multiplier: float,
) -> Tuple[Dict[str, float], PrivacyEngine]:
    fixed_model = ModuleValidator.fix(model)
    if fixed_model is not model:
        model = fixed_model

    model_params = {id(p) for p in model.parameters()}
    opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if model_params != opt_params:
        optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)

    model.train()
    privacy_engine = PrivacyEngine()

    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="functorch",
    )

    losses = []
    accs = []
    steps = 0

    stream = itertools.cycle(private_loader)
    for _ in range(warmup_steps):
        x, y = next(stream)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(max_steps):
        x, y = next(stream)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append((logits.argmax(dim=1) == y).float().mean().item())

        steps += 1
        if steps >= max_steps:
            break

    elapsed = time.perf_counter() - start
    return (
        {
            "train_loss": float(sum(losses) / max(len(losses), 1)),
            "train_acc": float(sum(accs) / max(len(accs), 1)),
            "seconds": elapsed,
            "steps": float(steps),
        },
        privacy_engine,
    )

