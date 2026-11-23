from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ab_test_showcase.config import ExperimentConfig


def _parse_revenue(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unexpected Revenue value: {value!r}")


def load_kaggle_baseline(path: str | Path) -> tuple[float, dict[str, float]]:
    """Read only historical session outcomes and visitor type for simulation calibration."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Kaggle CSV not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read Kaggle CSV {csv_path}: {exc}") from exc

    columns = {str(column).strip().casefold(): column for column in frame.columns}
    if "revenue" not in columns:
        raise ValueError("Kaggle CSV must contain a 'Revenue' column.")
    revenue = frame[columns["revenue"]].dropna().map(_parse_revenue)
    if revenue.empty:
        raise ValueError("Kaggle CSV has no non-null Revenue values.")
    baseline = float(revenue.mean())
    if not 0 < baseline < 1:
        raise ValueError("Revenue must include both purchasing and non-purchasing sessions.")

    visitor_shares = {"new": 0.4, "returning": 0.6}
    if "visitortype" in columns:
        visitor_type = frame.loc[revenue.index, columns["visitortype"]].astype(str).str.strip()
        normalized = visitor_type.str.casefold()
        new_count = int(normalized.isin({"new_visitor", "new visitor", "new"}).sum())
        returning_count = int(normalized.isin({"returning_visitor", "returning visitor", "returning"}).sum())
        valid_count = new_count + returning_count
        if valid_count:
            visitor_shares = {
                "new": new_count / valid_count,
                "returning": returning_count / valid_count,
            }
    return baseline, visitor_shares


def simulate_experiment(
    sample_size_per_arm: int,
    config: ExperimentConfig,
    baseline_conversion: float | None = None,
    visitor_shares: dict[str, float] | None = None,
) -> pd.DataFrame:
    if sample_size_per_arm < 2:
        raise ValueError("sample_size_per_arm must be at least 2.")
    baseline = baseline_conversion if baseline_conversion is not None else config.baseline_conversion
    if not 0 < baseline < 1:
        raise ValueError("baseline_conversion must be between 0 and 1.")
    if baseline + 1.2 * config.simulated_treatment_effect_absolute >= 1:
        raise ValueError("baseline_conversion plus simulated effect must be less than 1.")

    shares = visitor_shares or {"new": 0.4, "returning": 0.6}
    new_share = float(shares.get("new", 0))
    returning_share = float(shares.get("returning", 0))
    share_total = new_share + returning_share
    if new_share < 0 or returning_share < 0 or share_total <= 0:
        raise ValueError("visitor_shares must have positive, non-negative weights.")
    probabilities = [new_share / share_total, returning_share / share_total]

    rng = np.random.default_rng(config.seed)
    total_users = 2 * sample_size_per_arm
    variant = np.repeat(["control", "treatment"], sample_size_per_arm)
    rng.shuffle(variant)
    visitor_type = rng.choice(
        ["new", "returning"], size=total_users, p=probabilities
    )
    device = rng.choice(
        ["desktop", "mobile", "tablet"],
        size=total_users,
        p=[0.45, 0.45, 0.10],
    )

    is_treatment = variant == "treatment"
    effect_multiplier = np.where(visitor_type == "new", 1.2, 0.8666666667)
    conversion_probability = baseline + (
        is_treatment * config.simulated_treatment_effect_absolute * effect_multiplier
    )
    converted = rng.random(total_users) < conversion_probability
    checkout_started = converted | (rng.random(total_users) < 0.22)
    error_probability = np.where(
        is_treatment,
        config.checkout_error_rate_treatment,
        config.checkout_error_rate_control,
    )
    checkout_error = checkout_started & (rng.random(total_users) < error_probability)

    sigma = config.order_value_log_sigma
    log_mean = np.log(config.average_order_value) - sigma**2 / 2
    order_value = rng.lognormal(log_mean, sigma, size=total_users)
    revenue_per_user = np.where(converted, order_value, 0.0)

    return pd.DataFrame(
        {
            "user_id": [f"user_{index:07d}" for index in range(total_users)],
            "variant": variant,
            "visitor_type": visitor_type,
            "device": device,
            "checkout_started": checkout_started,
            "converted": converted,
            "checkout_error": checkout_error,
            "revenue_per_user": revenue_per_user,
        }
    )
