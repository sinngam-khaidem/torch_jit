import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrivacyReport:
    epsilon: float
    delta: float
    steps: int
    sample_rate: float
    noise_multiplier: float
    method: str


def epsilon_abadi_approx(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
) -> float:
    """
    Lightweight approximation used when external accountants are unavailable.

    epsilon ~= q * sqrt(2 * T * log(1/delta)) / sigma
    where q is sample rate, T is number of steps, sigma is noise multiplier.
    """
    if noise_multiplier <= 0:
        return float("inf")
    if steps <= 0 or sample_rate <= 0:
        return 0.0
    return float(sample_rate * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / noise_multiplier)


def epsilon_opacus_rdp(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
) -> Optional[float]:
    """Use Opacus RDP accountant when available."""
    try:
        from opacus.accountants import RDPAccountant
    except Exception:
        return None

    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    eps = accountant.get_epsilon(delta=delta)
    return float(eps)


def compute_epsilon(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
    prefer_opacus: bool = True,
) -> PrivacyReport:
    method = "abadi-approx"
    eps = None

    if prefer_opacus:
        eps = epsilon_opacus_rdp(
            steps=steps,
            sample_rate=sample_rate,
            noise_multiplier=noise_multiplier,
            delta=delta,
        )
        if eps is not None:
            method = "opacus-rdp"

    if eps is None:
        eps = epsilon_abadi_approx(
            steps=steps,
            sample_rate=sample_rate,
            noise_multiplier=noise_multiplier,
            delta=delta,
        )

    return PrivacyReport(
        epsilon=float(eps),
        delta=float(delta),
        steps=int(steps),
        sample_rate=float(sample_rate),
        noise_multiplier=float(noise_multiplier),
        method=method,
    )


if __name__ == "__main__":
    report = compute_epsilon(
        steps=1000,
        sample_rate=0.01,
        noise_multiplier=1.1,
        delta=1e-5,
        prefer_opacus=True,
    )
    print(report)
