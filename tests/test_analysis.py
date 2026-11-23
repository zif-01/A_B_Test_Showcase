import pandas as pd
import pytest

from ab_test_showcase.analysis import analyze_experiment, analyze_proportion, check_srm, holm_adjust


def test_proportion_analysis_returns_effect_and_interval():
    frame = pd.DataFrame(
        {
            "variant": ["control"] * 1000 + ["treatment"] * 1000,
            "converted": [True] * 100 + [False] * 900 + [True] * 130 + [False] * 870,
        }
    )

    result = analyze_proportion(frame, "converted")

    assert result["absolute_lift"] == pytest.approx(0.03)
    assert result["ci_low"] < result["absolute_lift"] < result["ci_high"]
    assert result["p_value"] < 0.05


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_srm_flags_obvious_imbalance():
    frame = pd.DataFrame({"variant": ["control"] * 900 + ["treatment"] * 100})

    assert not check_srm(frame)["passed"]


def test_full_analysis_stops_when_srm_fails():
    frame = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(1000)],
            "variant": ["control"] * 900 + ["treatment"] * 100,
            "checkout_started": [False] * 1000,
            "converted": [False] * 1000,
            "checkout_error": [False] * 1000,
            "revenue_per_user": [0.0] * 1000,
        }
    )

    with pytest.raises(ValueError, match="Sample ratio mismatch"):
        analyze_experiment(frame, 0.05, 1.0, 100, 1)


def test_full_analysis_includes_proxy_and_revenue_distribution_diagnostics():
    converted_control = [True] * 40 + [False] * 460
    converted_treatment = [True] * 50 + [False] * 450
    converted = converted_control + converted_treatment
    frame = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(1000)],
            "variant": ["control"] * 500 + ["treatment"] * 500,
            "checkout_started": [True] * 1000,
            "converted": converted,
            "checkout_error": [False] * 1000,
            "revenue_per_user": [
                (float(index % 20 + 1) if did_convert else 0.0)
                for index, did_convert in enumerate(converted)
            ],
        }
    )

    results = analyze_experiment(frame, 0.05, 1.0, 100, 7)

    assert "checkout_started" in results
    assert results["revenue_per_user"]["control_zero_fraction"] == pytest.approx(0.92)
