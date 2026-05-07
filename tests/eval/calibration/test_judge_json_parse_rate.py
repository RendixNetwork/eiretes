"""Eiretes-side judge JSON parse-rate gate tests."""

from __future__ import annotations

import pytest

from eiretes.eval.calibration.gate_result import GateResult
from eiretes.eval.calibration.json_parse_rate import (
    DEFAULT_PARSE_RATE_THRESHOLD,
    SWAP_THRESHOLD,
    JudgeJsonParseRateFixture,
    JudgeJsonParseRateGate,
    measure_judge_json_parse_rate,
)


pytestmark = pytest.mark.asyncio


def _ok_factory(name: str = "ok"):
    async def _call() -> dict:
        return {"outcome": "correct"}
    return JudgeJsonParseRateFixture(name=name, judge_call=_call)


def _fail_factory(name: str = "fail", err: str = "judge raised"):
    async def _call() -> dict:
        raise RuntimeError(err)
    return JudgeJsonParseRateFixture(name=name, judge_call=_call)


# -- Pass / marginal / fail thresholds ----------------------------------


async def test_all_succeed_returns_pass():
    fixtures = [_ok_factory(f"f{i}") for i in range(20)]
    result = await measure_judge_json_parse_rate(fixtures)
    assert result.status == "pass"
    assert result.measured_rate == 1.0
    assert result.n_samples == 20


async def test_marginal_zone():
    """93% pass rate: above SWAP_THRESHOLD, below default."""
    fixtures = [_ok_factory(f"ok{i}") for i in range(93)] + [
        _fail_factory(f"fail{i}") for i in range(7)
    ]
    result = await measure_judge_json_parse_rate(fixtures)
    assert result.measured_rate == 0.93
    assert SWAP_THRESHOLD <= result.measured_rate < DEFAULT_PARSE_RATE_THRESHOLD
    assert result.status == "marginal"


async def test_fail_zone():
    """80% pass rate: below SWAP_THRESHOLD."""
    fixtures = [_ok_factory(f"ok{i}") for i in range(80)] + [
        _fail_factory(f"fail{i}") for i in range(20)
    ]
    result = await measure_judge_json_parse_rate(fixtures)
    assert result.measured_rate == 0.8
    assert result.status == "fail"


# -- Failure recording --------------------------------------------------


async def test_failures_recorded_in_details():
    fixtures = [
        _ok_factory("ok-1"),
        _fail_factory("crashy", err="something blew up"),
        _ok_factory("ok-2"),
    ]
    result = await measure_judge_json_parse_rate(fixtures)
    assert result.measured_rate == pytest.approx(2 / 3)
    failures = result.details["failures"]
    assert len(failures) == 1
    assert failures[0]["fixture_name"] == "crashy"
    assert "something blew up" in (failures[0]["error"] or "")


async def test_empty_fixtures_returns_fail():
    result = await measure_judge_json_parse_rate([])
    assert result.status == "fail"
    assert result.n_samples == 0
    assert result.details["reason"] == "no_fixtures_provided"


# -- Operator-facing wrapper --------------------------------------------


async def test_gate_helper_runs():
    fixtures = [_ok_factory(f"f{i}") for i in range(10)]
    gate = JudgeJsonParseRateGate(fixtures, threshold=0.95)
    assert gate.threshold == 0.95
    assert gate.n_fixtures == 10
    result = await gate.run()
    assert isinstance(result, GateResult)
    assert result.status == "pass"


async def test_custom_threshold():
    """0.85 threshold → 90% pass rate clears it."""
    fixtures = [_ok_factory(f"ok{i}") for i in range(9)] + [_fail_factory("x")]
    result = await measure_judge_json_parse_rate(fixtures, threshold=0.85)
    assert result.status == "pass"
