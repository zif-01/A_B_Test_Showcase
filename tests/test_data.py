import pandas as pd
import pytest

from ab_test_showcase.config import ExperimentConfig
from ab_test_showcase.data import load_kaggle_baseline, simulate_experiment


@pytest.fixture
def config():
    return ExperimentConfig(
        seed=17,
        baseline_conversion=0.065,
        mde_absolute=0.01,
        simulated_treatment_effect_absolute=0.012,
        alpha=0.05,
        power=0.8,
        allocation_ratio=1.0,
        eligible_users_per_day=1500,
        minimum_duration_days=7,
        average_order_value=75.0,
        order_value_log_sigma=0.9,
        checkout_error_rate_control=0.025,
        checkout_error_rate_treatment=0.026,
        maximum_checkout_error_increase=0.005,
        bootstrap_iterations=100,
    )


def test_simulation_is_reproducible_and_balanced(config):
    first = simulate_experiment(500, config)
    second = simulate_experiment(500, config)

    pd.testing.assert_frame_equal(first, second)
    assert first["user_id"].is_unique
    assert first["variant"].value_counts().to_dict() == {
        "control": 500,
        "treatment": 500,
    }
    assert first.loc[first["converted"], "checkout_started"].all()
    assert first.loc[first["checkout_error"], "checkout_started"].all()


def test_kaggle_csv_is_used_for_baseline_only(tmp_path):
    csv_path = tmp_path / "online_shoppers_intention.csv"
    pd.DataFrame(
        {
            "Revenue": [True, False, False, True],
            "VisitorType": ["New_Visitor", "Returning_Visitor", "New_Visitor", "Returning_Visitor"],
            "PageValues": [1000, 0, 0, 500],
        }
    ).to_csv(csv_path, index=False)

    baseline, shares = load_kaggle_baseline(csv_path)

    assert baseline == 0.5
    assert shares == {"new": 0.5, "returning": 0.5}


def test_invalid_revenue_value_is_not_silently_ignored(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"Revenue": ["unknown"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unexpected Revenue"):
        load_kaggle_baseline(csv_path)
