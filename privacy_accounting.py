import math
from dataclasses import dataclass
from typing import Optional
from opacus.accountants import RDPAccountant


@dataclass
class PrivacyReport:
    epsilon: float
    delta: float
    steps: int
    sample_rate: float
    noise_multiplier: float
    method: str


def epsilon_opacus_rdp(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
) -> Optional[float]:
    """Use Opacus RDP accountant when available."""

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
) -> PrivacyReport:
    eps = None
    method = "opacus-rdp"
    eps = epsilon_opacus_rdp(
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
        noise_multiplier=0.9,
        delta=1e-5,
    )
    print(report)
