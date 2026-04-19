import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from layers import ModelSpec, build_model
from privacy_accounting import compute_epsilon
from torch_privacy import CustomPrivacyEngine, DPConfig
from utils import IMAGE_DATASETS, evaluate_accuracy, get_device, get_image_loader, set_seed


def train_non_private(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    accs: List[float] = []
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
        "train_loss": float(sum(losses) / max(len(losses), 1)),
        "train_acc": float(sum(accs) / max(len(accs), 1)),
        "seconds": elapsed,
        "steps": float(steps),
    }


def train_private(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int,
    use_compile: bool,
    max_grad_norm: float,
    noise_multiplier: float,
    vmap_chunk_size: int,
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

    start = time.perf_counter()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        stats = engine.dp_step(
            x,
            y,
            optimizer,
            loss_fn=F.cross_entropy,
            clip_mode="flat",
            use_compile=use_compile
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


def benchmark_dataset(
    dataset_name: str,
    model_names: List[str],
    batch_size: int,
    max_steps: int,
    lr: float,
    max_grad_norm: float,
    noise_multiplier: float,
    delta: float,
    vmap_chunk_size: int,
    device: str,
) -> List[Dict[str, object]]:
    spec = IMAGE_DATASETS[dataset_name]
    train_loader = get_image_loader(dataset_name, batch_size=batch_size, train=True, subset_size=batch_size * max_steps * 2)
    test_loader = get_image_loader(dataset_name, batch_size=batch_size, train=False, subset_size=batch_size * max_steps)

    dataset_size = len(train_loader.dataset)
    sample_rate = min(1.0, batch_size / max(dataset_size, 1))

    rows: List[Dict[str, object]] = []
    methods = ["non_private_sgd", "dpsgd_vmap", "dpsgd_vmap_compile"]

    for model_name in model_names:
        for method in methods:
            model = build_model(
                ModelSpec(
                    model_name=model_name,
                    num_classes=spec.num_classes,
                    in_channels=spec.in_channels,
                    image_size=spec.image_size,
                )
            ).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            if method == "non_private_sgd":
                train_stats = train_non_private(model, train_loader, optimizer, device, max_steps=max_steps)
                epsilon = 0.0
                accountant_method = "none"
            else:
                train_stats = train_private(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    max_steps=max_steps,
                    use_compile=(method == "dpsgd_vmap_compile"),
                    max_grad_norm=max_grad_norm,
                    noise_multiplier=noise_multiplier,
                    vmap_chunk_size=vmap_chunk_size,
                )
                report = compute_epsilon(
                    steps=int(train_stats["steps"]),
                    sample_rate=sample_rate,
                    noise_multiplier=noise_multiplier,
                    delta=delta,
                )
                epsilon = report.epsilon
                accountant_method = report.method

            test_acc = evaluate_accuracy(model, test_loader, device)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "method": method,
                    "seconds": round(train_stats["seconds"], 4),
                    "train_acc": round(train_stats["train_acc"], 4),
                    "test_acc": round(test_acc, 4),
                    "train_loss": round(train_stats["train_loss"], 4),
                    "epsilon": round(float(epsilon), 4),
                    "delta": delta,
                    "accountant": accountant_method,
                }
            )

    return rows


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "dataset",
        "model",
        "method",
        "seconds",
        "train_acc",
        "test_acc",
        "epsilon",
        "delta",
        "accountant",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row[h]) for h in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DP-SGD methods across image datasets and models.")
    parser.add_argument("--datasets", type=str, default="mnist,cifar10,fakedata")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--vmap-chunk-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    model_map: Dict[str, List[str]] = {
        "mnist": ["mlp", "cnn"],
        "cifar10": ["cnn", "vit"],
        "fakedata": ["cnn", "vit"],
    }

    rows: List[Dict[str, object]] = []
    for dataset_name in datasets:
        if dataset_name not in IMAGE_DATASETS:
            print(f"Skipping unsupported dataset: {dataset_name}")
            continue
        dataset_rows = benchmark_dataset(
            dataset_name=dataset_name,
            model_names=model_map.get(dataset_name, ["cnn"]),
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            lr=args.lr,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=args.noise_multiplier,
            delta=args.delta,
            vmap_chunk_size=args.vmap_chunk_size,
            device=device,
        )
        rows.extend(dataset_rows)

    print_table(rows)


if __name__ == "__main__":
    main()
