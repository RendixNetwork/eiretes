"""JSON parse-rate gate for eiretes' 3 judge roles.

The validator dispatches to ``/v1/judge/eval``, ``/v1/judge/pairwise``,
``/v1/judge/multi`` — each backed by ``zai-org/GLM-5.1-TEE`` via
Chutes. TEE-hosted models on prior subnet deployments
(Kimi-K2.5-TEE) had high JSON-malformation rates; eiretes-side
calibration verifies the new model's parse rate before flipping
production traffic.

Acceptance: ≥98% parse rate, ≥95% schema-valid. Below 90% the model
isn't ready and JSON-repair retry can't recover; swap the model.

Operator workflow: build a fixture set per judge role (representative
``(bundle, expected_answer/expected_claims, ...)`` triples), call
``measure_judge_json_parse_rate(judge, fixtures)`` with the real
judge instance, inspect the ``GateResult``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from eiretes.eval.calibration.gate_result import GateResult


_logger = logging.getLogger(__name__)


DEFAULT_PARSE_RATE_THRESHOLD = 0.98
SWAP_THRESHOLD = 0.90


@dataclass(frozen=True)
class JudgeJsonParseRateFixture:
    """One probe input.

    ``judge_call`` is an awaitable factory — the harness calls
    ``await fixture.judge_call()`` to invoke the judge with this
    fixture's specific inputs. Wrapping in a factory lets the
    fixture carry role-specific args (bundle, expected_answer, etc.)
    without the harness needing to know the role's signature.
    """

    name: str
    judge_call: Any  # Callable[[], Awaitable[Any]]


@dataclass(frozen=True)
class _SampleOutcome:
    fixture_name: str
    parsed: bool
    error: str | None


async def measure_judge_json_parse_rate(
    fixtures: Iterable[JudgeJsonParseRateFixture],
    *,
    threshold: float = DEFAULT_PARSE_RATE_THRESHOLD,
    name: str = "judge_json_parse_rate",
) -> GateResult:
    """Run every fixture once, return aggregate parse rate.

    Each fixture's ``judge_call`` is awaited; whatever it returns is
    treated as success (the judge module already validated the
    response). Exceptions raised by the judge are counted as
    parse failures.
    """
    samples: list[_SampleOutcome] = []
    for fixture in fixtures:
        try:
            await fixture.judge_call()
        except Exception as exc:
            samples.append(
                _SampleOutcome(
                    fixture_name=fixture.name,
                    parsed=False,
                    error=f"judge_error: {exc}",
                )
            )
            continue
        samples.append(
            _SampleOutcome(
                fixture_name=fixture.name, parsed=True, error=None,
            ),
        )

    n = len(samples)
    if n == 0:
        return GateResult(
            name=name,
            status="fail",
            measured_rate=0.0,
            threshold=threshold,
            n_samples=0,
            details={"reason": "no_fixtures_provided"},
        )

    n_parsed = sum(1 for s in samples if s.parsed)
    rate = n_parsed / n
    if rate >= threshold:
        status = "pass"
    elif rate >= SWAP_THRESHOLD:
        status = "marginal"
    else:
        status = "fail"

    return GateResult(
        name=name,
        status=status,
        measured_rate=rate,
        threshold=threshold,
        n_samples=n,
        details={
            "n_parsed": n_parsed,
            "n_failed": n - n_parsed,
            "swap_threshold": SWAP_THRESHOLD,
            "failures": [
                dataclasses.asdict(s) for s in samples if not s.parsed
            ],
        },
    )


class JudgeJsonParseRateGate:
    """Operator-facing harness wrapper."""

    def __init__(
        self,
        fixtures: Iterable[JudgeJsonParseRateFixture],
        *,
        threshold: float = DEFAULT_PARSE_RATE_THRESHOLD,
        name: str = "judge_json_parse_rate",
    ) -> None:
        self._fixtures = list(fixtures)
        self._threshold = threshold
        self._name = name

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def n_fixtures(self) -> int:
        return len(self._fixtures)

    async def run(self) -> GateResult:
        return await measure_judge_json_parse_rate(
            self._fixtures, threshold=self._threshold, name=self._name,
        )


__all__ = [
    "DEFAULT_PARSE_RATE_THRESHOLD",
    "JudgeJsonParseRateFixture",
    "JudgeJsonParseRateGate",
    "SWAP_THRESHOLD",
    "measure_judge_json_parse_rate",
]
