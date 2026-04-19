"""Benchmark VMAP/JIT-friendly recurrent and attention layers under DP-SGD."""

import argparse
import itertools
import time

import torch
import torch.nn.functional as F

from layers import ModelSpec, build_model
from privacy_accounting import compute_epsilon
from torch_privacy import CustomPrivacyEngine, DPConfig
from utils import evaluate_accuracy, get_device, get_synthetic_sequence_loader, set_seed


def train_non_private(model, loader, optimizer, device: str, max_steps: int):
    """
    Train the model without DP-SGD and return training stats.

    Args:
        - max_steps: Number of steps to train for (excludes warmup).
    """
    model.train()
    losses, accs = [], []
    steps = 0

    start = time.perf_counter()
    for x, y in loader:
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

    return {
        "train_loss": sum(losses) / max(len(losses), 1),
        "train_acc": sum(accs) / max(len(accs), 1),
        "seconds": elapsed,
        "steps": steps,
    }


def train_private(model, loader, optimizer, device: str, max_steps: int, compile_update: bool, max_grad_norm: float, noise_multiplier: float):
    """
    Args:
        - max_steps: Number of steps to train for (excludes warmup).
        - compile_update: Whether to JIT-compile the DP update step (requires PyTorch 2.4+).
        - max_grad_norm: Clipping norm for DP-SGD.
        - noise_multiplier: Noise multiplier for DP-SGD.
    """

    model.train()
    engine = CustomPrivacyEngine(
        model,
        DPConfig(max_grad_norm=max_grad_norm, noise_multiplier=noise_multiplier, vmap_chunk_size=32),
    )

    losses, accs = [], []
    steps = 0

    start = time.perf_counter()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        stats = engine.dp_step(x, y, optimizer, loss_fn=F.cross_entropy, clip_mode="flat", use_compile=compile_update)

        losses.append(stats["loss"])
        accs.append(stats["acc"])

        steps += 1
        if steps >= max_steps:
            break
    elapsed = time.perf_counter() - start

    return {
        "train_loss": sum(losses) / max(len(losses), 1),
        "train_acc": sum(accs) / max(len(accs), 1),
        "seconds": elapsed,
        "steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark recurrent and attention layers for DP-SGD.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seq-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    train_loader = get_synthetic_sequence_loader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.seq_dim,
        num_classes=args.num_classes,
        size=args.batch_size * args.max_steps * 3,
        train=True,
    )
    test_loader = get_synthetic_sequence_loader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.seq_dim,
        num_classes=args.num_classes,
        size=args.batch_size * args.max_steps,
        train=False,
    )


    methods = ["non_private_sgd", "dpsgd_vmap", "dpsgd_vmap_compile"]
    model_names = ["rnn", "lstm", "mha"]

    print("model\tmethod\tsteady_seconds\ttrain_acc\ttest_acc\tepsilon\tsteady_samples_per_sec")

    for model_name in model_names:
        for method in methods:
            model = build_model(
                ModelSpec(
                    model_name=model_name,
                    num_classes=args.num_classes,
                    seq_input_dim=args.seq_dim,
                    seq_hidden_dim=args.hidden_dim,
                    seq_len=args.seq_len,
                )
            ).to(device)

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

            if method == "non_private_sgd":
                stats = train_non_private(model, train_loader, optimizer, device, max_steps=args.max_steps)
                epsilon = 0.0
            else:
                stats = train_private(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    max_steps=args.max_steps,
                    compile_update=(method == "dpsgd_vmap_compile"),
                    max_grad_norm=args.max_grad_norm,
                    noise_multiplier=args.noise_multiplier,
                )
                report = compute_epsilon(
                    steps=int(stats["steps"]),
                    sample_rate=args.batch_size / len(train_loader.dataset),
                    noise_multiplier=args.noise_multiplier,
                    delta=args.delta,
                )
                epsilon = report.epsilon



if __name__ == "__main__":
    main()
