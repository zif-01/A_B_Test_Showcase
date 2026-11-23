from __future__ import annotations

import math
from statistics import NormalDist


def required_sample_size_per_arm(
    baseline: float,
    absolute_mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate equal-arm sample size for a two-sided test of two proportions."""
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1.")
    if not 0 < absolute_mde < 1 or baseline + absolute_mde >= 1:
        raise ValueError("baseline + absolute_mde must be less than 1.")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between 0 and 1.")

    treatment_rate = baseline + absolute_mde
    pooled_rate = (baseline + treatment_rate) / 2
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    numerator = (
        z_alpha * math.sqrt(2 * pooled_rate * (1 - pooled_rate))
        + z_power
        * math.sqrt(
            baseline * (1 - baseline)
            + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2
    return math.ceil(numerator / absolute_mde**2)


def estimate_duration_days(
    sample_size_per_arm: int,
    eligible_users_per_day: int,
    minimum_duration_days: int = 7,
) -> int:
    if sample_size_per_arm <= 0 or eligible_users_per_day <= 0 or minimum_duration_days <= 0:
        raise ValueError("Sample size, traffic, and minimum duration must be positive.")
    needed_days = math.ceil(2 * sample_size_per_arm / eligible_users_per_day)
    return max(needed_days, minimum_duration_days)
