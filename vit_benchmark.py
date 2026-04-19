import argparse
import itertools
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from layers import ModelSpec, build_model
from privacy_accounting import compute_epsilon
from torch_privacy import CustomPrivacyEngine, DPConfig
from utils import IMAGE_DATASETS, evaluate_accuracy, get_device, get_image_loader, set_seed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT DP-SGD benchmark.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "fakedata"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
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

    data_spec = IMAGE_DATASETS[args.dataset]
    train_loader = get_image_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=True,
        subset_size=args.batch_size * args.max_steps * 3,
    )
    test_loader = get_image_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=False,
        subset_size=args.batch_size * args.max_steps,
        shuffle=False,
    )

    methods = [
        "dpsgd_vmap",
        "dpsgd_vmap_compile",
    ]

    print("method\ttotal_seconds\tsteady_seconds\ttrain_acc\ttest_acc\tepsilon\tsteady_samples_per_sec")

    for method in methods:
        model = build_model(
            ModelSpec(
                model_name="vit",
                num_classes=data_spec.num_classes,
                in_channels=data_spec.in_channels,
                image_size=data_spec.image_size,
            )
        ).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        engine = CustomPrivacyEngine(
            model,
            DPConfig(max_grad_norm=args.max_grad_norm, noise_multiplier=args.noise_multiplier, vmap_chunk_size=None),
        )

        losses: List[float] = []
        accs: List[float] = []
        steps = 0

        stream = itertools.cycle(train_loader)

        total_start = time.perf_counter()
        for _ in range(args.warmup_steps):
            x, y = next(stream)
            x = x.to(device)
            y = y.to(device)

            if method == "dpsgd_vmap":
                stats = engine.dp_step(x, y, optimizer, loss_fn=F.cross_entropy, use_compile=False)
            else:
                stats = engine.dp_step(x, y, optimizer, loss_fn=F.cross_entropy, use_compile=True)

        steady_start = time.perf_counter()
        for _ in range(args.max_steps):
            x, y = next(stream)
            x = x.to(device)
            y = y.to(device)

            if method == "dpsgd_vmap":
                stats = engine.dp_step(x, y, optimizer, loss_fn=F.cross_entropy, use_compile=False)
            else:
                stats = engine.dp_step(x, y, optimizer, loss_fn=F.cross_entropy, use_compile=True)

            losses.append(stats["loss"])
            accs.append(stats["acc"])
            steps += 1

        steady_elapsed = time.perf_counter() - steady_start
        total_elapsed = time.perf_counter() - total_start
        test_acc = evaluate_accuracy(model, test_loader, device)
        report = compute_epsilon(
            steps=steps + args.warmup_steps,
            sample_rate=args.batch_size / len(train_loader.dataset),
            noise_multiplier=args.noise_multiplier,
            delta=args.delta,
        )
        steady_samples_per_sec = (steps * args.batch_size) / max(steady_elapsed, 1e-9)

        print(
            f"{method}\t{total_elapsed:.4f}\t{steady_elapsed:.4f}\t{(sum(accs)/max(len(accs),1)):.4f}\t"
            f"{test_acc:.4f}\t{report.epsilon:.4f}\t{steady_samples_per_sec:.2f}"
        )


if __name__ == "__main__":
    main()
