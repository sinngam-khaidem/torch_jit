"""Explore alternative DP-SGD tweaks: clipping variants and Poisson subsampling."""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from layers import ModelSpec, build_model
from privacy_accounting import compute_epsilon
from torch_privacy import CustomPrivacyEngine, DPConfig
from utils import IMAGE_DATASETS, get_device, get_image_loader, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alternative DP-SGD experiments (clipping + Poisson subsampling).")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "fakedata"])
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "vit"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    ds = IMAGE_DATASETS[args.dataset]
    model_choices = [
        ("flat", False),
        ("layerwise", False),
        ("flat", True),
        ("layerwise", True),
    ]

    train_loader = get_image_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=True,
        subset_size=args.batch_size * args.max_steps * 3,
    )

    print("clip_mode\tpoisson\tseconds\tavg_loss\tavg_acc\tepsilon")

    for clip_mode, poisson in model_choices:
        model = build_model(
            ModelSpec(
                model_name=args.model,
                in_channels=ds.in_channels,
                image_size=ds.image_size,
                num_classes=ds.num_classes,
            )
        ).to(device)

        engine = CustomPrivacyEngine(
            model,
            DPConfig(
                max_grad_norm=args.max_grad_norm,
                noise_multiplier=args.noise_multiplier,
                poisson_sampling=poisson,
                sample_rate=args.sample_rate,
            ),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        losses: List[float] = []
        accs: List[float] = []
        steps = 0

        start = time.perf_counter()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            stats = engine.dp_step(
                x,
                y,
                optimizer,
                loss_fn=F.cross_entropy,
                clip_mode=clip_mode,
                use_compile=False,
                poisson_sampling=poisson,
                sample_rate=args.sample_rate,
            )
            losses.append(stats["loss"])
            accs.append(stats["acc"])

            steps += 1
            if steps >= args.max_steps:
                break

        elapsed = time.perf_counter() - start
        effective_sample_rate = args.sample_rate if poisson else args.batch_size / len(train_loader.dataset)
        report = compute_epsilon(
            steps=steps,
            sample_rate=effective_sample_rate,
            noise_multiplier=args.noise_multiplier,
            delta=args.delta,
        )

        print(
            f"{clip_mode}\t{poisson}\t{elapsed:.4f}\t{(sum(losses)/max(len(losses),1)):.4f}\t"
            f"{(sum(accs)/max(len(accs),1)):.4f}\t{report.epsilon:.4f}"
        )


if __name__ == "__main__":
    main()
