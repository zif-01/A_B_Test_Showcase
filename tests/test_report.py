from ab_test_showcase.report import decide


def test_rollout_requires_effect_above_mde_and_safe_guardrail():
    conversion = {"ci_low": 0.012, "ci_high": 0.025}
    checkout_error = {"ci_high": 0.004}

    recommendation, _ = decide(conversion, checkout_error, 0.01, 0.005)

    assert recommendation == "Внедрять поэтапно"


def test_uncertain_guardrail_blocks_rollout():
    conversion = {"ci_low": 0.012, "ci_high": 0.025}
    checkout_error = {"ci_high": 0.006}

    recommendation, _ = decide(conversion, checkout_error, 0.01, 0.005)

    assert recommendation == "Не масштабировать; проверить guardrail"
