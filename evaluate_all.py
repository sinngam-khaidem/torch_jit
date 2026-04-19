import argparse
import itertools
import time
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from jax_models import jax_available, train_jax_dp
from custom_models import ModelSpec, build_model
from opacus_runner import train_opacus
from opacus_models import build_opacus_model
from privacy_accounting import compute_epsilon
from torch_privacy import CustomPrivacyEngine, DPConfig
from utils import IMAGE_DATASETS, evaluate_accuracy, get_device, get_image_loader, get_synthetic_sequence_loader, set_seed
import warnings
warnings.filterwarnings("ignore")

def train_non_private(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int,
    warmup_steps: int,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    accs: List[float] = []
    steps = 0

    stream = itertools.cycle(loader)
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
    return {
        "train_loss": float(sum(losses) / max(len(losses), 1)),
        "train_acc": float(sum(accs) / max(len(accs), 1)),
        "seconds": elapsed,
        "steps": float(steps),
    }


def train_ours(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int,
    warmup_steps: int,
    max_grad_norm: float,
    noise_multiplier: float,
    vmap_chunk_size: Optional[int],
    use_compile: bool,
) -> Dict[str, float]:
    model.train()
    engine = CustomPrivacyEngine(
        model,
        DPConfig(
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            vmap_chunk_size=vmap_chunk_size,
        ),
    )

    losses: List[float] = []
    accs: List[float] = []
    steps = 0

    stream = itertools.cycle(loader)
    for _ in range(warmup_steps):
        x, y = next(stream)
        x = x.to(device)
        y = y.to(device)
        engine.dp_step(x, y, optimizer, loss_fn=F.cross_entropy, use_compile=use_compile)

    start = time.perf_counter()
    for _ in range(max_steps):
        x, y = next(stream)
        x = x.to(device)
        y = y.to(device)
        stats = engine.dp_step(
            x,
            y,
            optimizer,
            loss_fn=F.cross_entropy,
            clip_mode="flat",
            use_compile=use_compile,
        )

        losses.append(stats["loss"])
        accs.append(stats["acc"])

        steps += 1
        if steps >= max_steps:
            break

    elapsed = time.perf_counter() - start
    return {
        "train_loss": float(sum(losses) / max(len(losses), 1)),
        "train_acc": float(sum(accs) / max(len(accs), 1)),
        "seconds": elapsed,
        "steps": float(steps),
    }


def print_table(rows: List[Dict[str, object]]) -> None:
    console = Console()
    table = Table(title="DP-SGD Full Evaluation", show_lines=False)

    table.add_column("Dataset", style="bright_cyan", no_wrap=True)
    table.add_column("Model", style="bright_magenta")
    table.add_column("Method", style="yellow")
    table.add_column("Seconds", justify="right", style="green")
    table.add_column("Train Acc", justify="right", style="bright_green")
    table.add_column("Test Acc", justify="right", style="bright_green")
    table.add_column("Epsilon", justify="right", style="bright_blue")
    table.add_column("Delta", justify="right", style="blue")
    table.add_column("Accountant", style="white")

    for row in rows:
        table.add_row(
            str(row["dataset"]),
            str(row["model"]),
            str(row["method"]),
            str(row["seconds"]),
            str(row["train_acc"]),
            str(row["test_acc"]),
            str(row["epsilon"]),
            str(row["delta"]),
            str(row["accountant"]),
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Non-DP, JAX, Opacus, and Ours across datasets.")
    parser.add_argument("--datasets", type=str, default="mnist,cifar10,synthetic_sequence")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--vmap-chunk-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-ours-compile", action="store_false", dest="ours_compile", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    rows: List[Dict[str, object]] = []
    for dataset_name in datasets:
        if dataset_name == "synthetic_sequence":
            train_loader = get_synthetic_sequence_loader(
                batch_size=args.batch_size,
                seq_len=32,
                input_dim=64,
                num_classes=10,
                size=args.batch_size * args.max_steps * 3,
                train=True,
            )
            test_loader = get_synthetic_sequence_loader(
                batch_size=args.batch_size,
                seq_len=32,
                input_dim=64,
                num_classes=10,
                size=args.batch_size * args.max_steps,
                train=False,
            )
            model_names = ["rnn", "lstm", "gru", "mha"]
            dataset_spec = ModelSpec(
                model_name="rnn",
                num_classes=10,
                seq_input_dim=64,
                seq_hidden_dim=128,
                seq_len=32,
            )
            sample_rate = min(1.0, args.batch_size / max(len(train_loader.dataset), 1))
        else:
            if dataset_name not in IMAGE_DATASETS:
                print(f"Skipping unsupported dataset: {dataset_name}")
                continue
            image_spec = IMAGE_DATASETS[dataset_name]
            train_loader = get_image_loader(
                dataset_name,
                batch_size=args.batch_size,
                train=True,
                subset_size=args.batch_size * args.max_steps * 3,
            )
            test_loader = get_image_loader(
                dataset_name,
                batch_size=args.batch_size,
                train=False,
                subset_size=args.batch_size * args.max_steps,
                shuffle=False,
            )
            model_names = ["mlp", "cnn", "vit"]
            dataset_spec = ModelSpec(
                model_name="mlp",
                num_classes=image_spec.num_classes,
                in_channels=image_spec.in_channels,
                image_size=image_spec.image_size,
            )
            sample_rate = min(1.0, args.batch_size / max(len(train_loader.dataset), 1))

        for model_name in model_names:
            for method in ["non_dpsgd", "jax", "opacus", "ours"]:
                if method == "jax" and not jax_available():
                    raise SystemExit("JAX is not available. Install jax and jaxlib to run this method.")

                if method == "jax":
                    jax_stats = train_jax_dp(
                        model_name=model_name,
                        spec=ModelSpec(
                            model_name=model_name,
                            num_classes=dataset_spec.num_classes,
                            in_channels=dataset_spec.in_channels,
                            image_size=dataset_spec.image_size,
                            seq_input_dim=dataset_spec.seq_input_dim,
                            seq_hidden_dim=dataset_spec.seq_hidden_dim,
                            seq_len=dataset_spec.seq_len,
                        ),
                        train_loader=train_loader,
                        test_loader=test_loader,
                        max_steps=args.max_steps,
                        warmup_steps=args.warmup_steps,
                        lr=args.lr,
                        max_grad_norm=args.max_grad_norm,
                        noise_multiplier=args.noise_multiplier,
                    )
                    report = compute_epsilon(
                        steps=int(jax_stats["steps"]),
                        sample_rate=sample_rate,
                        noise_multiplier=args.noise_multiplier,
                        delta=args.delta,
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "method": "jax",
                            "seconds": round(jax_stats["seconds"], 4),
                            "train_acc": round(jax_stats["train_acc"], 4),
                            "test_acc": round(jax_stats["test_acc"], 4),
                            "epsilon": round(float(report.epsilon), 4),
                            "delta": args.delta,
                            "accountant": report.method,
                        }
                    )
                    continue

                model_spec = ModelSpec(
                    model_name=model_name,
                    num_classes=dataset_spec.num_classes,
                    in_channels=dataset_spec.in_channels,
                    image_size=dataset_spec.image_size,
                    seq_input_dim=dataset_spec.seq_input_dim,
                    seq_hidden_dim=dataset_spec.seq_hidden_dim,
                    seq_len=dataset_spec.seq_len,
                )

                if method == "opacus":
                    torch_model = build_opacus_model(model_spec).to(device)
                else:
                    torch_model = build_model(model_spec).to(device)

                optimizer = torch.optim.SGD(torch_model.parameters(), lr=args.lr)

                if method == "non_dpsgd":
                    train_stats = train_non_private(
                        torch_model,
                        train_loader,
                        optimizer,
                        device,
                        max_steps=args.max_steps,
                        warmup_steps=args.warmup_steps,
                    )
                    epsilon = 0.0
                    accountant_method = "none"
                elif method == "opacus":
                    try:
                        train_stats, privacy_engine = train_opacus(
                            torch_model,
                            train_loader,
                            optimizer,
                            device,
                            max_steps=args.max_steps,
                            warmup_steps=args.warmup_steps,
                            max_grad_norm=args.max_grad_norm,
                            noise_multiplier=args.noise_multiplier,
                        )
                        epsilon = privacy_engine.get_epsilon(delta=args.delta)
                        accountant_method = "opacus-privacy-engine"
                    except ValueError as exc:
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method,
                                "seconds": "n/a",
                                "train_acc": "n/a",
                                "test_acc": "n/a",
                                "epsilon": "n/a",
                                "delta": args.delta,
                                "accountant": f"opacus-unsupported: {exc}",
                            }
                        )
                        continue
                else:
                    train_stats = train_ours(
                        torch_model,
                        train_loader,
                        optimizer,
                        device,
                        max_steps=args.max_steps,
                        warmup_steps=args.warmup_steps,
                        max_grad_norm=args.max_grad_norm,
                        noise_multiplier=args.noise_multiplier,
                        vmap_chunk_size=args.vmap_chunk_size,
                        use_compile=args.ours_compile,
                    )
                    report = compute_epsilon(
                        steps=int(train_stats["steps"]),
                        sample_rate=sample_rate,
                        noise_multiplier=args.noise_multiplier,
                        delta=args.delta,
                    )
                    epsilon = report.epsilon
                    accountant_method = report.method

                test_acc = evaluate_accuracy(torch_model, test_loader, device)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": method,
                        "seconds": round(train_stats["seconds"], 4),
                        "train_acc": round(train_stats["train_acc"], 4),
                        "test_acc": round(test_acc, 4),
                        "epsilon": round(float(epsilon), 4),
                        "delta": args.delta,
                        "accountant": accountant_method,
                    }
                )

    print_table(rows)


if __name__ == "__main__":
    main()
