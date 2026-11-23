from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ab_test_showcase.analysis import analyze_experiment
from ab_test_showcase.config import load_config
from ab_test_showcase.data import load_kaggle_baseline, simulate_experiment
from ab_test_showcase.design import estimate_duration_days, required_sample_size_per_arm
from ab_test_showcase.report import save_effect_plot, save_json, write_memo

ROOT = Path(__file__).resolve().parents[2]


def run(
    config_path: str | Path = ROOT / "config" / "experiment.yaml",
    output_dir: str | Path = ROOT / "reports",
    kaggle_csv: str | Path | None = None,
) -> dict[str, Any]:
    config, dataset = load_config(config_path)
    csv_path = kaggle_csv or dataset.get("kaggle_csv")
    visitor_shares = None
    source_description = (
        f"демонстрационный baseline {config.baseline_conversion:.2%}; "
        "Kaggle-файл не передан"
    )
    if csv_path:
        baseline, visitor_shares = load_kaggle_baseline(csv_path)
        config = replace(config, baseline_conversion=baseline)
        config.validate()
        source_description = (
            f"baseline сессий из {Path(csv_path).name} ({baseline:.2%}); "
            "данные не рандомизированы"
        )

    sample_size = required_sample_size_per_arm(
        config.baseline_conversion,
        config.mde_absolute,
        config.alpha,
        config.power,
    )
    duration = estimate_duration_days(
        sample_size,
        config.eligible_users_per_day,
        config.minimum_duration_days,
    )
    design = {
        "baseline_conversion": config.baseline_conversion,
        "treatment_rate_at_mde": config.baseline_conversion + config.mde_absolute,
        "mde_absolute": config.mde_absolute,
        "alpha": config.alpha,
        "power": config.power,
        "allocation_ratio": config.allocation_ratio,
        "sample_size_per_arm": sample_size,
        "total_sample_size": 2 * sample_size,
        "eligible_users_per_day": config.eligible_users_per_day,
        "estimated_duration_days": duration,
        "minimum_duration_days": config.minimum_duration_days,
        "stopping_rule": "Fixed sample size and at least the minimum full-week duration; no unplanned peeking.",
    }

    frame = simulate_experiment(
        sample_size,
        config,
        baseline_conversion=config.baseline_conversion,
        visitor_shares=visitor_shares,
    )
    results = analyze_experiment(
        frame,
        alpha=config.alpha,
        allocation_ratio=config.allocation_ratio,
        bootstrap_iterations=config.bootstrap_iterations,
        seed=config.seed,
    )

    output = Path(output_dir)
    payload = {
        "design": design,
        "results": results,
        "config": asdict(config),
        "source_description": source_description,
        "simulated_treatment_effect_absolute": config.simulated_treatment_effect_absolute,
    }
    save_json(output / "results.json", payload)
    save_effect_plot(
        results,
        output / "figures" / "effect_summary.png",
        config.mde_absolute,
        config.maximum_checkout_error_increase,
    )
    write_memo(
        output / "business_memo.md",
        results,
        design,
        {
            "mde_absolute": config.mde_absolute,
            "alpha": config.alpha,
            "power": config.power,
            "simulated_treatment_effect_absolute": config.simulated_treatment_effect_absolute,
            "maximum_checkout_error_increase": config.maximum_checkout_error_increase,
        },
        source_description,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible A/B test showcase.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "experiment.yaml"),
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports"),
        help="Directory for JSON results, memo, and figure.",
    )
    parser.add_argument(
        "--kaggle-csv",
        help="Optional local Online Shoppers Purchasing Intention CSV used for baseline calibration.",
    )
    args = parser.parse_args()
    payload = run(args.config, args.output_dir, args.kaggle_csv)
    conversion = payload["results"]["conversion"]
    print(
        json.dumps(
            {
                "sample_size_per_arm": payload["design"]["sample_size_per_arm"],
                "estimated_duration_days": payload["design"]["estimated_duration_days"],
                "absolute_conversion_lift": conversion["absolute_lift"],
                "conversion_p_value": conversion["p_value"],
                "srm_passed": payload["results"]["srm"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
