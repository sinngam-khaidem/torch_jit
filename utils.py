"""Shared utilities for data loading, reproducibility, and evaluation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass
class DatasetSpec:
    name: str
    num_classes: int
    in_channels: int
    image_size: int


IMAGE_DATASETS: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(name="mnist", num_classes=10, in_channels=1, image_size=28),
    "cifar10": DatasetSpec(name="cifar10", num_classes=10, in_channels=3, image_size=32),
    "fakedata": DatasetSpec(name="fakedata", num_classes=10, in_channels=3, image_size=32),
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(explicit_device: Optional[str] = None) -> str:
    if explicit_device is not None:
        return explicit_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_image_dataset(
    dataset_name: str,
    train: bool,
    root: str = "./data",
    download: bool = True,
):
    name = dataset_name.lower()
    if name not in IMAGE_DATASETS:
        raise ValueError(f"Unsupported image dataset: {dataset_name}")

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        return torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transform)

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        return torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)

    # FakeData fallback dataset
    spec = IMAGE_DATASETS[name]
    transform = transforms.Compose([transforms.ToTensor()])
    size = 10_000 if train else 2_000
    return torchvision.datasets.FakeData(
        size=size,
        image_size=(spec.in_channels, spec.image_size, spec.image_size),
        num_classes=spec.num_classes,
        transform=transform,
    )


def get_image_loader(
    dataset_name: str,
    batch_size: int,
    train: bool = True,
    subset_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    root: str = "./data",
    download: bool = True,
) -> DataLoader:
    shuffle = train if shuffle is None else shuffle

    try:
        dataset = _build_image_dataset(dataset_name, train=train, root=root, download=download)
    except Exception:
        # Graceful fallback if downloads are unavailable.
        dataset = _build_image_dataset("fakedata", train=train, root=root, download=False)

    if subset_size is not None and subset_size < len(dataset):
        dataset = Subset(dataset, list(range(subset_size)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_mnist_loader(cfg) -> Tuple[DataLoader, DataLoader]:
    trainloader = get_image_loader(
        dataset_name="mnist",
        batch_size=cfg.batch_size,
        train=True,
        subset_size=getattr(cfg, "train_subset", None),
    )
    testloader = get_image_loader(
        dataset_name="mnist",
        batch_size=cfg.batch_size,
        train=False,
        subset_size=getattr(cfg, "test_subset", None),
        shuffle=False,
    )
    return trainloader, testloader


def get_synthetic_sequence_loader(
    batch_size: int,
    seq_len: int = 64,
    input_dim: int = 64,
    num_classes: int = 10,
    size: int = 4096,
    train: bool = True,
) -> DataLoader:
    """
    Generates a synthetic sequence dataset and wrap it in a DataLoader.
    """
    g = torch.Generator().manual_seed(123 if train else 321)
    x = torch.randn(size, seq_len, input_dim, generator=g)

    # Correlate labels with signal in first few features for learnability.
    logits = x[:, :, : min(input_dim, num_classes)].mean(dim=1)
    y = logits.argmax(dim=1) % num_classes

    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=train)


def evaluate_accuracy(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: str,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return float(correct / max(total, 1))


def now() -> float:
    return time.perf_counter()


__all__ = [
    "DatasetSpec",
    "IMAGE_DATASETS",
    "set_seed",
    "get_device",
    "get_image_loader",
    "get_mnist_loader",
    "get_synthetic_sequence_loader",
    "evaluate_accuracy",
    "now",
]


if __name__ == "__main__":
    class Config:
        batch_size = 32
        train_subset = 512
        test_subset = 256

    cfg = Config()
    trainloader, testloader = get_mnist_loader(cfg)
    print(f"Train batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")
