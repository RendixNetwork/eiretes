"""Pairwise position-bias chi-square test.

The validator's ``_judge_miner`` randomizes A/B assignment per task
via ``secrets.randbelow(2)``. This gate verifies the empirical
distribution of A/B slot assignments matches the expected uniform
50/50 over a sample. A 200-sample chi-square test with α=0.05 has
ample power to detect a biased RNG (which would silently re-introduce
position bias into pairwise scoring).

This is a pure-local test — no LLM calls. The operator runs it once
per validator deployment to verify nothing is wrong with the system
RNG (an exotic but possible failure mode in TEE / containerized
environments).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from eiretes.eval.calibration.gate_result import GateResult


# Chi-square critical values for 1 degree of freedom.
# α=0.05 → 3.841; α=0.01 → 6.635.
DEFAULT_POSITION_BIAS_ALPHA = 0.05
_CHI_SQUARE_CRITICAL_VALUES_DF1 = {
    0.01: 6.635,
    0.05: 3.841,
    0.10: 2.706,
}


def measure_pairwise_position_bias(
    samples: Sequence[Literal["A", "B"]],
    *,
    alpha: float = DEFAULT_POSITION_BIAS_ALPHA,
    name: str = "pairwise_position_bias",
) -> GateResult:
    """Chi-square goodness-of-fit test against uniform A/B distribution.

    ``samples`` is the sequence of miner-position assignments observed
    over N calls. Pass: chi-square statistic below the critical value
    for ``alpha`` and 1 degree of freedom (cannot reject the null
    "uniform 50/50"). Fail: above critical → biased RNG.

    Returns ``measured_rate = chi_square_statistic`` (NOT a percentage)
    and ``threshold = critical_value`` so the operator can read the
    test directly. ``details["a_count"]`` / ``details["b_count"]`` /
    ``details["expected"]`` carry the underlying counts.
    """
    if alpha not in _CHI_SQUARE_CRITICAL_VALUES_DF1:
        raise ValueError(
            f"alpha={alpha} not supported; pick one of "
            f"{sorted(_CHI_SQUARE_CRITICAL_VALUES_DF1.keys())}"
        )
    critical = _CHI_SQUARE_CRITICAL_VALUES_DF1[alpha]
    n = len(samples)
    if n == 0:
        return GateResult(
            name=name,
            status="fail",
            measured_rate=0.0,
            threshold=critical,
            n_samples=0,
            details={"reason": "no_samples_provided"},
        )

    a_count = sum(1 for s in samples if s == "A")
    b_count = sum(1 for s in samples if s == "B")
    if a_count + b_count != n:
        return GateResult(
            name=name,
            status="fail",
            measured_rate=0.0,
            threshold=critical,
            n_samples=n,
            details={
                "reason": "samples_must_be_A_or_B",
                "a_count": a_count,
                "b_count": b_count,
                "n_invalid": n - (a_count + b_count),
            },
        )

    expected = n / 2
    chi_square = (
        ((a_count - expected) ** 2 + (b_count - expected) ** 2) / expected
    )

    # Pass if chi-square is below the critical value (cannot reject H0).
    status = "pass" if chi_square < critical else "fail"
    return GateResult(
        name=name,
        status=status,
        measured_rate=chi_square,
        threshold=critical,
        n_samples=n,
        details={
            "a_count": a_count,
            "b_count": b_count,
            "expected_per_slot": expected,
            "alpha": alpha,
            "df": 1,
            "a_fraction": a_count / n,
            "b_fraction": b_count / n,
        },
    )


__all__ = [
    "DEFAULT_POSITION_BIAS_ALPHA",
    "measure_pairwise_position_bias",
]
