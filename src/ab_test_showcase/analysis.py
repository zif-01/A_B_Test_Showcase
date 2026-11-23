from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _wilson_interval(events: int, total: int, alpha: float) -> tuple[float, float]:
    if total == 0:
        raise ValueError("Cannot calculate a proportion for an empty group.")
    z = stats.norm.ppf(1 - alpha / 2)
    proportion = events / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def analyze_proportion(
    frame: pd.DataFrame,
    outcome: str,
    alpha: float = 0.05,
) -> dict[str, float | int | str]:
    control = frame.loc[frame["variant"] == "control", outcome].astype(bool)
    treatment = frame.loc[frame["variant"] == "treatment", outcome].astype(bool)
    n_control, n_treatment = len(control), len(treatment)
    if not n_control or not n_treatment:
        raise ValueError("Both control and treatment groups must be non-empty.")

    events_control, events_treatment = int(control.sum()), int(treatment.sum())
    rate_control, rate_treatment = events_control / n_control, events_treatment / n_treatment
    difference = rate_treatment - rate_control

    pooled = (events_control + events_treatment) / (n_control + n_treatment)
    expected = [
        n_control * pooled,
        n_control * (1 - pooled),
        n_treatment * pooled,
        n_treatment * (1 - pooled),
    ]
    if min(expected) < 5:
        _, p_value = stats.fisher_exact(
            [
                [events_treatment, n_treatment - events_treatment],
                [events_control, n_control - events_control],
            ],
            alternative="two-sided",
        )
        method = "Fisher exact test"
    elif pooled in {0, 1}:
        p_value = 1.0
        method = "Two-proportion z-test"
    else:
        standard_error = math.sqrt(
            pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment)
        )
        z_score = difference / standard_error
        p_value = float(2 * stats.norm.sf(abs(z_score)))
        method = "Two-proportion z-test"

    control_low, control_high = _wilson_interval(events_control, n_control, alpha)
    treatment_low, treatment_high = _wilson_interval(events_treatment, n_treatment, alpha)
    ci_low = difference - math.sqrt(
        (rate_control - control_low) ** 2
        + (treatment_high - rate_treatment) ** 2
    )
    ci_high = difference + math.sqrt(
        (control_high - rate_control) ** 2
        + (rate_treatment - treatment_low) ** 2
    )
    return {
        "outcome": outcome,
        "control_n": n_control,
        "treatment_n": n_treatment,
        "control_events": events_control,
        "treatment_events": events_treatment,
        "control_rate": rate_control,
        "treatment_rate": rate_treatment,
        "absolute_lift": difference,
        "relative_lift": difference / rate_control if rate_control else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_value),
        "method": method,
    }


def check_srm(frame: pd.DataFrame, allocation_ratio: float = 1.0) -> dict[str, float | int | bool]:
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be positive.")
    if not set(frame["variant"].dropna().unique()) <= {"control", "treatment"}:
        raise ValueError("variant must contain only 'control' and 'treatment'.")
    control_count = int((frame["variant"] == "control").sum())
    treatment_count = int((frame["variant"] == "treatment").sum())
    if not control_count or not treatment_count:
        raise ValueError("SRM requires non-empty control and treatment groups.")
    expected_control = (control_count + treatment_count) / (1 + allocation_ratio)
    expected_treatment = expected_control * allocation_ratio
    statistic, p_value = stats.chisquare(
        [control_count, treatment_count],
        f_exp=[expected_control, expected_treatment],
    )
    return {
        "control_count": control_count,
        "treatment_count": treatment_count,
        "p_value": float(p_value),
        "chi_square": float(statistic),
        "passed": bool(p_value >= 0.001),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be between 0 and 1.")
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        corrected = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, corrected)
        adjusted[int(index)] = running_max
    return adjusted


def analyze_revenue(
    frame: pd.DataFrame,
    alpha: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, float | int | str]:
    control = frame.loc[frame["variant"] == "control", "revenue_per_user"].to_numpy(float)
    treatment = frame.loc[frame["variant"] == "treatment", "revenue_per_user"].to_numpy(float)
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Revenue analysis requires at least two users per group.")

    mean_control, mean_treatment = float(control.mean()), float(treatment.mean())
    difference = mean_treatment - mean_control
    variance_control, variance_treatment = control.var(ddof=1), treatment.var(ddof=1)
    standard_error = math.sqrt(variance_control / len(control) + variance_treatment / len(treatment))
    degrees_freedom = (
        (variance_control / len(control) + variance_treatment / len(treatment)) ** 2
        / (
            (variance_control / len(control)) ** 2 / (len(control) - 1)
            + (variance_treatment / len(treatment)) ** 2 / (len(treatment) - 1)
        )
    )
    t_statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    critical = stats.t.ppf(1 - alpha / 2, degrees_freedom)
    welch_low = difference - critical * standard_error
    welch_high = difference + critical * standard_error

    rng = np.random.default_rng(seed)
    bootstrap_differences: list[np.ndarray] = []
    remaining = bootstrap_iterations
    while remaining:
        batch_size = min(100, remaining)
        control_indices = rng.integers(0, len(control), size=(batch_size, len(control)))
        treatment_indices = rng.integers(0, len(treatment), size=(batch_size, len(treatment)))
        bootstrap_differences.append(
            treatment[treatment_indices].mean(axis=1)
            - control[control_indices].mean(axis=1)
        )
        remaining -= batch_size
    bootstrap = np.concatenate(bootstrap_differences)
    bootstrap_low, bootstrap_high = np.quantile(
        bootstrap, [alpha / 2, 1 - alpha / 2]
    )
    return {
        "control_mean": mean_control,
        "treatment_mean": mean_treatment,
        "control_zero_fraction": float(np.mean(control == 0)),
        "treatment_zero_fraction": float(np.mean(treatment == 0)),
        "control_skewness": float(stats.skew(control, bias=False)) if variance_control else None,
        "treatment_skewness": float(stats.skew(treatment, bias=False)) if variance_treatment else None,
        "absolute_lift": difference,
        "welch_ci_low": float(welch_low),
        "welch_ci_high": float(welch_high),
        "p_value": float(p_value),
        "welch_t_statistic": float(t_statistic),
        "bootstrap_ci_low": float(bootstrap_low),
        "bootstrap_ci_high": float(bootstrap_high),
        "bootstrap_iterations": bootstrap_iterations,
        "method": "Welch t-test and percentile bootstrap",
    }


def analyze_subgroups(
    frame: pd.DataFrame,
    variables: Sequence[str],
    alpha: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for variable in variables:
        if variable not in frame.columns:
            continue
        for segment, subgroup in frame.groupby(variable, observed=True):
            if subgroup["variant"].nunique() != 2:
                continue
            result = analyze_proportion(subgroup, "converted", alpha)
            results.append(
                {
                    "variable": variable,
                    "segment": str(segment),
                    **result,
                }
            )
    adjusted = holm_adjust([float(row["p_value"]) for row in results])
    for row, adjusted_p in zip(results, adjusted, strict=True):
        row["holm_adjusted_p_value"] = adjusted_p
    return results


def analyze_experiment(
    frame: pd.DataFrame,
    alpha: float,
    allocation_ratio: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    if frame["user_id"].duplicated().any():
        raise ValueError("Expected one analysis row per randomized user.")
    if not set(frame["variant"].dropna().unique()) <= {"control", "treatment"}:
        raise ValueError("variant must contain only 'control' and 'treatment'.")
    if frame[
        [
            "variant",
            "checkout_started",
            "converted",
            "checkout_error",
            "revenue_per_user",
        ]
    ].isna().any().any():
        raise ValueError("Experiment outcomes contain missing values.")
    if (
        frame["converted"].astype(bool) & ~frame["checkout_started"].astype(bool)
    ).any():
        raise ValueError("A converted user must have started checkout.")
    if (frame["revenue_per_user"] < 0).any():
        raise ValueError("Revenue per user cannot be negative.")
    if (frame["revenue_per_user"].gt(0) != frame["converted"].astype(bool)).any():
        raise ValueError("Revenue must be positive for converted users and zero otherwise.")
    srm = check_srm(frame, allocation_ratio)
    if not srm["passed"]:
        raise ValueError(
            f"Sample ratio mismatch detected (p={srm['p_value']:.4g}); "
            "do not interpret metric results."
        )
    return {
        "srm": srm,
        "checkout_started": analyze_proportion(frame, "checkout_started", alpha),
        "conversion": analyze_proportion(frame, "converted", alpha),
        "checkout_error": analyze_proportion(frame, "checkout_error", alpha),
        "revenue_per_user": analyze_revenue(
            frame, alpha, bootstrap_iterations, seed + 1
        ),
        "subgroups": analyze_subgroups(
            frame, ["visitor_type", "device"], alpha
        ),
    }
