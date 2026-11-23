from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    baseline_conversion: float
    mde_absolute: float
    simulated_treatment_effect_absolute: float
    alpha: float
    power: float
    allocation_ratio: float
    eligible_users_per_day: int
    minimum_duration_days: int
    average_order_value: float
    order_value_log_sigma: float
    checkout_error_rate_control: float
    checkout_error_rate_treatment: float
    maximum_checkout_error_increase: float
    bootstrap_iterations: int

    def validate(self) -> None:
        for name in ("baseline_conversion", "mde_absolute", "simulated_treatment_effect_absolute"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.baseline_conversion + self.mde_absolute >= 1:
            raise ValueError("baseline_conversion + mde_absolute must be less than 1.")
        if self.baseline_conversion + 1.2 * self.simulated_treatment_effect_absolute >= 1:
            raise ValueError("baseline_conversion plus the maximum simulated treatment effect must be less than 1.")
        if not 0 < self.alpha < 1 or not 0 < self.power < 1:
            raise ValueError("alpha and power must be between 0 and 1.")
        if self.allocation_ratio != 1.0:
            raise ValueError("This showcase currently supports 1:1 allocation only.")
        if self.eligible_users_per_day <= 0 or self.minimum_duration_days <= 0:
            raise ValueError("Traffic and minimum duration must be positive.")
        if self.average_order_value <= 0 or self.order_value_log_sigma <= 0:
            raise ValueError("Order value and log-scale sigma must be positive.")
        for name in ("checkout_error_rate_control", "checkout_error_rate_treatment"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if not 0 <= self.maximum_checkout_error_increase <= 1:
            raise ValueError("maximum_checkout_error_increase must be between 0 and 1.")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100.")


def load_config(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment"), dict):
        raise ValueError("Configuration must contain an 'experiment' mapping.")

    try:
        experiment = ExperimentConfig(**raw["experiment"])
    except TypeError as exc:
        raise ValueError(f"Invalid experiment configuration: {exc}") from exc
    experiment.validate()

    dataset = raw.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("'dataset' configuration must be a mapping.")
    return experiment, dataset
