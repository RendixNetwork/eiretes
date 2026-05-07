"""The composite is the product of five factors; verify each independently.

Multiplicative shape closes sandbagging — a candidate can't trade off
a wrong answer for strong tool-use to game the score.
"""
from __future__ import annotations

import pytest

from eiretes.eval.composite import composite_score
from eiretes.eval.models import EvalOutcome


def _outcome(name: str = "correct") -> EvalOutcome:
    return EvalOutcome(outcome=name, failure_mode=None, guidance="")


def test_correct_no_tool_no_must_not_claim_yields_one():
    score = composite_score(
        outcome=_outcome("correct"),
        candidate_response="The answer is 42.",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=100,
        cost_usd=0.001,
    )
    assert score.composite == pytest.approx(1.0, rel=1e-6)
    assert score.outcome_score == 1.0
    assert score.tool_attestation_factor == 1.0
    assert score.efficiency_factor == 1.0
    assert score.hallucination_knockout == 1.0
    assert score.cost_attestation_knockout == 1.0
    assert score.knockout_reason is None


def test_wrong_outcome_zeros_composite():
    score = composite_score(
        outcome=_outcome("wrong"),
        candidate_response="anything",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=10,
        cost_usd=0.001,
    )
    assert score.composite == 0.0


def test_required_tool_missing_zeros_composite():
    """A miner who skipped or faked the required tool gets 0."""
    score = composite_score(
        outcome=_outcome("correct"),
        candidate_response="The article says X.",
        must_not_claim=[],
        required_tool="web_search",
        ledger_tools=[],  # nothing in ledger
        latency_ms=10,
        cost_usd=0.001,
    )
    assert score.tool_attestation_factor == 0.0
    assert score.composite == 0.0
    assert score.knockout_reason is not None
    assert "web_search" in score.knockout_reason


def test_required_tool_matches_prefix_in_ledger():
    """``web_search`` matches ``web_search.open_page`` etc."""
    score = composite_score(
        outcome=_outcome("correct"),
        candidate_response="ok",
        must_not_claim=[],
        required_tool="web_search",
        ledger_tools=["web_search.open_page"],
        latency_ms=10,
        cost_usd=0.001,
    )
    assert score.tool_attestation_factor == 1.0


def test_must_not_claim_match_zeros_composite():
    score = composite_score(
        outcome=_outcome("correct"),
        candidate_response="The capital of France is London.",
        must_not_claim=["london"],  # case-insensitive match
        required_tool=None,
        ledger_tools=[],
        latency_ms=10,
        cost_usd=0.001,
    )
    assert score.hallucination_knockout == 0.0
    assert score.composite == 0.0
    assert "london" in (score.knockout_reason or "").lower()


def test_zero_cost_floors_composite():
    """A turn with sub-floor cost is suspicious — the miner cached or faked."""
    score = composite_score(
        outcome=_outcome("correct"),
        candidate_response="ok",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=10,
        cost_usd=0.0,
        cost_floor_usd=0.00005,
    )
    assert score.cost_attestation_knockout == 0.0
    assert score.composite == 0.0
    assert "suspicious_zero_cost" in (score.knockout_reason or "")


def test_efficiency_full_within_50_percent_budget():
    score = composite_score(
        outcome=_outcome("correct"),
        candidate_response="ok",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=4000,
        cost_usd=0.001,
        latency_budget_ms=10_000,
        cost_budget_usd=0.01,
    )
    assert score.efficiency_factor == 1.0
    assert score.composite == 1.0


def test_efficiency_drops_to_half_at_full_budget():
    score = composite_score(
        outcome=_outcome("correct"),
        candidate_response="ok",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=10_000,
        cost_usd=0.001,
        latency_budget_ms=10_000,
        cost_budget_usd=0.01,
    )
    assert score.efficiency_factor == 0.5
    # composite = 1 * 1 * 0.5 * 1 * 1 = 0.5
    assert score.composite == pytest.approx(0.5, rel=1e-6)


def test_partial_outcome_scores_half():
    score = composite_score(
        outcome=_outcome("partial"),
        candidate_response="The answer is something.",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=10,
        cost_usd=0.001,
    )
    assert score.outcome_score == 0.5
    assert score.composite == pytest.approx(0.5, rel=1e-6)


def test_disputed_treated_as_partial():
    score = composite_score(
        outcome=_outcome("disputed"),
        candidate_response="The answer is X (orracle says Y).",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=10,
        cost_usd=0.001,
    )
    assert score.outcome_score == 0.5


def test_hallucinated_outcome_zeros_composite():
    score = composite_score(
        outcome=_outcome("hallucinated"),
        candidate_response="anything",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=10,
        cost_usd=0.001,
    )
    assert score.outcome_score == 0.0
    assert score.composite == 0.0
    assert score.knockout_reason == "outcome=hallucinated"
