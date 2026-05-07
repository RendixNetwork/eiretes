"""Pairwise position-bias chi-square test."""

from __future__ import annotations

import secrets

import pytest

from eiretes.eval.calibration.position_bias import (
    measure_pairwise_position_bias,
)


# -- Statistical pass / fail behavior -----------------------------------


def test_perfectly_uniform_passes():
    """Exactly 100/100 split — chi-square = 0 → pass."""
    samples = ["A", "B"] * 100
    result = measure_pairwise_position_bias(samples)
    assert result.status == "pass"
    assert result.measured_rate == 0.0
    assert result.n_samples == 200


def test_slightly_uneven_still_passes_under_alpha_0_05():
    """52/48 split: chi-square = (52-50)^2/50 + (48-50)^2/50
    = 2*4/50 = 0.16 < 3.841 → pass at α=0.05."""
    samples = ["A"] * 52 + ["B"] * 48
    result = measure_pairwise_position_bias(samples)
    assert result.status == "pass"
    assert result.measured_rate == pytest.approx(0.16, rel=0.01)


def test_obviously_biased_fails():
    """80/20 split: chi-square = (80-50)^2/50 + (20-50)^2/50
    = 18 + 18 = 36 ≫ critical → fail."""
    samples = ["A"] * 80 + ["B"] * 20
    result = measure_pairwise_position_bias(samples)
    assert result.status == "fail"
    assert result.measured_rate > 30.0


def test_real_secrets_randbelow_passes_at_200_samples():
    """secrets.randbelow(2) — the validator's actual A/B picker —
    should produce a uniform-enough distribution to pass the gate
    on every reasonable random seed. Run with 200 samples (validator
    plan target). Use ~10 trials of 200 to keep flake risk small."""
    failures = 0
    for _ in range(10):
        samples = ["A" if secrets.randbelow(2) == 0 else "B" for _ in range(200)]
        result = measure_pairwise_position_bias(samples)
        if result.status != "pass":
            failures += 1
    # Under H0 (true uniform), false-positive rate at α=0.05 is 5%.
    # 10 trials → expected 0.5 failures; we tolerate up to 3 to make
    # the test robust against random unlucky sequences.
    assert failures <= 3, (
        f"secrets.randbelow(2) failed {failures}/10 trials of 200; "
        f"may indicate a biased RNG"
    )


# -- Edge cases ---------------------------------------------------------


def test_empty_samples_returns_fail():
    result = measure_pairwise_position_bias([])
    assert result.status == "fail"
    assert result.n_samples == 0


def test_invalid_label_returns_fail():
    """Samples must be 'A' or 'B' — anything else is rejected."""
    samples = ["A", "B", "X"]
    result = measure_pairwise_position_bias(samples)
    assert result.status == "fail"
    assert result.details["reason"] == "samples_must_be_A_or_B"
    assert result.details["n_invalid"] == 1


def test_alpha_0_01_is_stricter():
    """Moderately uneven distribution that passes at α=0.05 still
    passes at α=0.01 (critical = 6.635). For a definitive fail,
    the chi-square must exceed the higher critical value."""
    # 60/40 → chi-square = (60-50)^2/50 + (40-50)^2/50 = 2 + 2 = 4
    # critical_0.05 = 3.841 → 4.0 > 3.841 → fail at α=0.05
    # critical_0.01 = 6.635 → 4.0 < 6.635 → pass at α=0.01
    samples = ["A"] * 60 + ["B"] * 40
    result_05 = measure_pairwise_position_bias(samples, alpha=0.05)
    assert result_05.status == "fail"
    result_01 = measure_pairwise_position_bias(samples, alpha=0.01)
    assert result_01.status == "pass"


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        measure_pairwise_position_bias(["A", "B"], alpha=0.123)


def test_details_contain_fractions():
    samples = ["A"] * 105 + ["B"] * 95
    result = measure_pairwise_position_bias(samples)
    assert result.details["a_count"] == 105
    assert result.details["b_count"] == 95
    assert result.details["a_fraction"] == 0.525
    assert result.details["b_fraction"] == 0.475
    assert result.details["alpha"] == 0.05
    assert result.details["df"] == 1
