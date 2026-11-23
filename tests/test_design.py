import pytest

from ab_test_showcase.design import estimate_duration_days, required_sample_size_per_arm


def test_sample_size_increases_when_mde_gets_smaller():
    smaller_effect = required_sample_size_per_arm(0.065, 0.008)
    larger_effect = required_sample_size_per_arm(0.065, 0.012)

    assert smaller_effect > larger_effect


def test_duration_respects_full_week_and_traffic():
    assert estimate_duration_days(1000, 1000, minimum_duration_days=7) == 7
    assert estimate_duration_days(8000, 1000, minimum_duration_days=7) == 16


def test_invalid_mde_is_rejected():
    with pytest.raises(ValueError, match="baseline \\+ absolute_mde"):
        required_sample_size_per_arm(0.95, 0.10)
