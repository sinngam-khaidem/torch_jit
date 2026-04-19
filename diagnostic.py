import argparse
import time
from typing import Dict

import torch
import torch.nn.functional as F

from layers import ModelSpec, build_model
from torch_privacy import CustomPrivacyEngine, DPConfig, per_sample_grads_loop, per_sample_grads_microbatch
from utils import get_device, set_seed


def max_abs_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    diffs = []
    for key in a:
        diffs.append((a[key] - b[key]).abs().max().item())
    return float(max(diffs) if diffs else 0.0)


def run_once(model_name: str, batch_size: int, seq_len: int, seq_dim: int, device: str, microbatch_size: int) -> None:
    if model_name in {"rnn", "lstm", "gru", "mha"}:
        model = build_model(
            ModelSpec(model_name=model_name, num_classes=10, seq_input_dim=seq_dim, seq_len=seq_len, seq_hidden_dim=128)
        ).to(device)
        x = torch.randn(batch_size, seq_len, seq_dim, device=device)
    elif model_name == "vit":
        model = build_model(ModelSpec(model_name="vit", in_channels=3, image_size=32)).to(device)
        x = torch.randn(batch_size, 3, 32, 32, device=device)
    elif model_name == "cnn":
        model = build_model(ModelSpec(model_name="cnn", in_channels=1, image_size=28)).to(device)
        x = torch.randn(batch_size, 1, 28, 28, device=device)
    else:
        model = build_model(ModelSpec(model_name="mlp", in_channels=1, image_size=28)).to(device)
        x = torch.randn(batch_size, 1, 28, 28, device=device)

    y = torch.randint(0, 10, (batch_size,), device=device)

    results = {}

    t0 = time.perf_counter()
    loop_grads = per_sample_grads_loop(model, x, y, loss_fn=F.cross_entropy)
    results["loop"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    micro_grads = per_sample_grads_microbatch(
        model,
        x,
        y,
        microbatch_size=microbatch_size,
        loss_fn=F.cross_entropy,
    )
    results["microbatch"] = time.perf_counter() - t0

    engine = CustomPrivacyEngine(model, DPConfig(max_grad_norm=1.0, noise_multiplier=0.0, vmap_chunk_size=None))
    t0 = time.perf_counter()
    _, _, vmap_grads = engine.make_per_sample_gradients(x, y, loss_fn=F.cross_entropy)
    results["vmap"] = time.perf_counter() - t0

    rows = [
        (
            "loop",
            results["loop"],
            0.0,
        ),
        (
            "microbatch",
            results["microbatch"],
            max_abs_diff(loop_grads, micro_grads),
        ),
        (
            "vmap",
            results["vmap"],
            max_abs_diff(loop_grads, vmap_grads),
        ),
    ]

    try:
        import opacus  # noqa: F401

        opacus_note = "available"
    except Exception:
        opacus_note = "not installed (comparison skipped)"

    print(f"model={model_name} batch_size={batch_size} device={device}")
    print(f"opacus={opacus_note}")
    print("method\tseconds\tmax_abs_diff_vs_loop")
    for method, seconds, diff in rows:
        print(f"{method}\t{seconds:.6f}\t{diff:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-sample gradient techniques.")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "rnn", "lstm", "gru", "mha", "vit"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seq-dim", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    run_once(
        model_name=args.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seq_dim=args.seq_dim,
        device=device,
        microbatch_size=args.microbatch_size,
    )


if __name__ == "__main__":
    main()
