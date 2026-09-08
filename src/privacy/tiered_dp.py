"""
Core novelty logic: maps privacy tier -> DP noise multiplier.

High noise = tighter privacy (more protection, less utility).
Low noise = looser privacy (less protection, more utility).

stigma_sensitive gets the tight budget; non_sensitive gets the loose budget.
This is the actual contribution of the project -- everything else in the
pipeline (FL, plain DP) is standard technique; this file is where "risk-aware"
becomes a concrete number.
"""

NOISE_MULTIPLIER_STIGMA_SENSITIVE = 1.0   # tight privacy budget
NOISE_MULTIPLIER_NON_SENSITIVE = 0.3      # loose privacy budget

MAX_GRAD_NORM = 1.0  # L2 clipping threshold, same for both tiers
DELTA = 1e-5          # standard DP delta for epsilon accounting


def noise_for_tier(tier: str) -> float:
    if tier == "stigma_sensitive":
        return NOISE_MULTIPLIER_STIGMA_SENSITIVE
    return NOISE_MULTIPLIER_NON_SENSITIVE