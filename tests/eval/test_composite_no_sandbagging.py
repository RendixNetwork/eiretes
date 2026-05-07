"""Sandbagging-resistance: a fast/cheap wrong answer cannot beat a slow correct one.

The multiplicative composite makes outcome dominate. A miner who
sandbagged the answer cannot recover by being efficient.
"""
from __future__ import annotations

import pytest

from eiretes.eval.composite import composite_score
from eiretes.eval.models import EvalOutcome


def test_fast_wrong_loses_to_slow_correct():
    """Wrong outcome × any efficiency = 0 < correct × any efficiency."""
    fast_wrong = composite_score(
        outcome=EvalOutcome(outcome="wrong"),
        candidate_response="quick wrong answer",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=50,
        cost_usd=0.0001,
        latency_budget_ms=10_000,
        cost_budget_usd=0.01,
    )
    slow_correct = composite_score(
        outcome=EvalOutcome(outcome="correct"),
        candidate_response="thorough correct answer",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=9_000,
        cost_usd=0.005,
        latency_budget_ms=10_000,
        cost_budget_usd=0.01,
    )
    assert fast_wrong.composite < slow_correct.composite
    assert fast_wrong.composite == 0.0


def test_required_tool_skipped_loses_to_slow_grounded_use():
    """A miner who skips the required tool to be fast scores 0."""
    fast_skipper = composite_score(
        outcome=EvalOutcome(outcome="correct"),  # judge fooled by surface form
        candidate_response="answer without tool use",
        must_not_claim=[],
        required_tool="web_search",
        ledger_tools=[],  # ledger says no tool calls
        latency_ms=50,
        cost_usd=0.0001,
        latency_budget_ms=10_000,
        cost_budget_usd=0.01,
    )
    honest_user = composite_score(
        outcome=EvalOutcome(outcome="correct"),
        candidate_response="answer grounded in search",
        must_not_claim=[],
        required_tool="web_search",
        ledger_tools=["web_search"],
        latency_ms=8_000,
        cost_usd=0.004,
        latency_budget_ms=10_000,
        cost_budget_usd=0.01,
    )
    assert fast_skipper.composite == 0.0
    assert honest_user.composite > 0.4


def test_zero_cost_short_circuit_cant_win():
    """A miner who returns instantly with no LLM spend is suspicious."""
    suspicious = composite_score(
        outcome=EvalOutcome(outcome="correct"),
        candidate_response="instant cached answer",
        must_not_claim=[],
        required_tool=None,
        ledger_tools=[],
        latency_ms=1,
        cost_usd=0.0,
        cost_floor_usd=0.00005,
        latency_budget_ms=10_000,
        cost_budget_usd=0.01,
    )
    assert suspicious.composite == 0.0
    assert suspicious.cost_attestation_knockout == 0.0
