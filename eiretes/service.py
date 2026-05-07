"""Eiretes judge service.

Endpoints:
    GET  /healthz                   — liveness
    POST /v1/judge/pairwise         — pairwise preference (A vs B)
    POST /v1/judge/multi            — outer-metric judge (grounded /
                                       retrieval / safety) in one call
    POST /v1/judge/eval             — reference-based LLM-as-judge
    POST /v1/judge/eval/composite   — pure-function composite (no LLM)

Per-miner feedback retrieval lives on owner-api directly (hotkey-signed
``GET /v1/eval/feedback``); eiretes is purely the judge service.

The validator engine in eirel-ai posts (prompt, answer_a, answer_b) to
``/v1/judge/pairwise`` for the dominant 0.40-weight signal in per-task
scoring. ``/v1/judge/eval`` is the reference-based scorer used for
``grounded_correctness`` in multi-metric scoring (separate from
pairwise). Position-bias defense is the validator's responsibility:
call pairwise twice with A/B swapped and average the two scores.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from eiretes.eval.bundle import JudgeInputBundle
from eiretes.eval.composite import composite_score as _eval_composite_score
from eiretes.eval.judge import EvalJudge
from eiretes.eval.multi_judge import MultiJudge, MultiJudgeVerdict
from eiretes.eval.pairwise import PairwiseJudge, PairwiseVerdict
from eiretes.eval.models import (
    EvalCompositeScore,
    EvalOutcome,
    FailureMode,
    OracleSource,
    Outcome,
)
from eiretes.utils import int_env

_logger = logging.getLogger(__name__)


app = FastAPI(title="Eiretes Eval Judge Service")


# -- /healthz --------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "judge_model": os.getenv("EIREL_EVAL_JUDGE_MODEL", "local-rubric-judge"),
        "rubric_version": "eval_judge_v1",
    }


# -- POST /v1/judge/eval ---------------------------------------------------


class EvalJudgeRequest(BaseModel):
    """Single-task judge call.

    The validator engine in eirel-ai builds the ``bundle`` (task-shape
    fields + candidate response) and passes per-call extras —
    ``expected_answer``, ``must_not_claim``, ``required_tool``,
    ``oracle_source`` — alongside. Eiretes is a pure judge: it doesn't
    fetch, render, or dispatch anything.
    """

    bundle: JudgeInputBundle
    expected_answer: str = Field(min_length=1)
    must_not_claim: list[str] = Field(default_factory=list)
    required_tool: str | None = Field(default=None, max_length=64)
    oracle_source: OracleSource = Field(default="deterministic")


class EvalJudgeResponse(BaseModel):
    outcome: Outcome
    failure_mode: FailureMode | None = None
    guidance: str = ""


@app.post("/v1/judge/eval", response_model=EvalJudgeResponse)
async def judge_eval(body: EvalJudgeRequest) -> EvalJudgeResponse:
    """LLM-as-judge call for one prepared task.

    Validator-driven: the validator engine fetches the task via the
    existing claim flow, dispatches the prompt to the candidate, then
    posts the (bundle, candidate response in bundle.answers, per-call
    extras) here for grading.
    """
    judge = EvalJudge()
    try:
        outcome = await judge.judge(
            bundle=body.bundle,
            expected_answer=body.expected_answer,
            must_not_claim=list(body.must_not_claim),
            required_tool=body.required_tool,
            oracle_source=body.oracle_source,
        )
    finally:
        await judge.aclose()
    return EvalJudgeResponse(
        outcome=outcome.outcome,
        failure_mode=outcome.failure_mode,
        guidance=outcome.guidance,
    )


# -- POST /v1/judge/pairwise -----------------------------------------------


class PairwiseJudgeRequest(BaseModel):
    """Pairwise preference judge call.

    ``bundle.answers`` MUST be ``[answer_a, answer_b]``. The validator
    runs this twice per task with A/B swapped (or once with random
    A/B assignment) for position-bias defense.

    ``expected_answer`` (optional) is the consensus / deterministic
    gold the judge anchors correctness on. When provided, the judge
    flips from "no factuality assumed" to "use this as the
    correctness anchor" and rewards correct + well-phrased over
    incorrect + well-phrased. Empty / null = legacy behavior.
    """

    bundle: JudgeInputBundle
    expected_answer: str | None = None


class PairwiseJudgeResponse(BaseModel):
    winner: str  # "A" | "B" | "tie"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    category_scores: dict[str, dict[str, int]] | None = None


@app.post("/v1/judge/pairwise", response_model=PairwiseJudgeResponse)
async def judge_pairwise(body: PairwiseJudgeRequest) -> PairwiseJudgeResponse:
    """Pairwise preference judge for the 0.40-weight per-task signal.

    Returns ``{winner, confidence, reason, category_scores}``. The judge
    sees only what the bundle's ``dispatch_for("pairwise")`` includes —
    question, answer_a/b, optional constraints + attached_summary +
    conversation. No raw documents, no tool traces, no expected answer.
    """
    judge = PairwiseJudge()
    try:
        verdict: PairwiseVerdict = await judge.judge(
            bundle=body.bundle,
            expected_answer=body.expected_answer,
        )
    finally:
        await judge.aclose()
    category_scores_payload: dict[str, dict[str, int]] | None = None
    if verdict.category_scores is not None:
        category_scores_payload = {
            field: {"A": pair["A"], "B": pair["B"]}
            for field, pair in verdict.category_scores.model_dump().items()
            if isinstance(pair, dict) and "A" in pair and "B" in pair
        }
    return PairwiseJudgeResponse(
        winner=verdict.winner,
        confidence=verdict.confidence,
        reason=verdict.reason,
        category_scores=category_scores_payload,
    )


# -- POST /v1/judge/multi --------------------------------------------------


class MultiJudgeRequest(BaseModel):
    """Outer-metric judge call (grounded / retrieval / safety).

    ``bundle.answers`` MUST be ``[candidate_response]``. Per-call
    extras (``expected_answer``, ``candidate_citations``,
    ``applicable_metrics``) ride alongside the bundle.
    """

    bundle: JudgeInputBundle
    expected_answer: str | None = None
    candidate_citations: list[str] = Field(default_factory=list)
    applicable_metrics: list[str] = Field(default_factory=list)


class MultiDimensionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class MultiJudgeResponse(BaseModel):
    grounded_correctness: MultiDimensionScore | None = None
    retrieval_quality: MultiDimensionScore | None = None
    instruction_safety: MultiDimensionScore | None = None


@app.post("/v1/judge/multi", response_model=MultiJudgeResponse)
async def judge_multi(body: MultiJudgeRequest) -> MultiJudgeResponse:
    """Score the candidate along the outer dimensions in one LLM call.

    The dimensions are independent — pairwise preference is judged
    separately by ``/v1/judge/pairwise`` against the OpenAI baseline.
    """
    judge = MultiJudge()
    try:
        verdict: MultiJudgeVerdict = await judge.judge(
            bundle=body.bundle,
            expected_answer=body.expected_answer,
            candidate_citations=list(body.candidate_citations),
            applicable_metrics=list(body.applicable_metrics),
        )
    finally:
        await judge.aclose()
    return MultiJudgeResponse(
        grounded_correctness=(
            MultiDimensionScore(
                score=verdict.grounded_correctness.score,
                rationale=verdict.grounded_correctness.rationale,
            )
            if verdict.grounded_correctness is not None else None
        ),
        retrieval_quality=(
            MultiDimensionScore(
                score=verdict.retrieval_quality.score,
                rationale=verdict.retrieval_quality.rationale,
            )
            if verdict.retrieval_quality is not None else None
        ),
        instruction_safety=(
            MultiDimensionScore(
                score=verdict.instruction_safety.score,
                rationale=verdict.instruction_safety.rationale,
            )
            if verdict.instruction_safety is not None else None
        ),
    )


# -- POST /v1/judge/eval/composite -----------------------------------------


class EvalCompositeRequest(BaseModel):
    """Compute the multiplicative composite given an outcome + attestation."""

    outcome: Outcome
    failure_mode: FailureMode | None = None
    candidate_response: str = ""
    must_not_claim: list[str] = Field(default_factory=list)
    required_tool: str | None = None
    ledger_tools: list[str] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_budget_ms: int | None = Field(default=None, ge=0)
    cost_budget_usd: float | None = Field(default=None, ge=0.0)
    cost_floor_usd: float | None = Field(default=None, ge=0.0)
    # Multi-judge outer-dimension scores. ``None`` → N/A for this
    # task type → gate bypassed (composite not zeroed by missing
    # dimensions).
    grounded_correctness_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
    )
    instruction_safety_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
    )
    # Miner-perspective pairwise win-rate vs the OpenAI baseline.
    # Adds ±0.10 bonus on top of outcome_score; ``None`` → 0 bonus.
    pairwise_preference_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
    )


@app.post(
    "/v1/judge/eval/composite", response_model=EvalCompositeScore,
)
async def judge_eval_composite(body: EvalCompositeRequest) -> EvalCompositeScore:
    """Pure-function composite (multiplicative + knockouts). No LLM call."""
    outcome = EvalOutcome(
        outcome=body.outcome,
        failure_mode=body.failure_mode,
        guidance="",
    )
    return _eval_composite_score(
        outcome=outcome,
        candidate_response=body.candidate_response,
        must_not_claim=list(body.must_not_claim),
        required_tool=body.required_tool,
        ledger_tools=list(body.ledger_tools),
        latency_ms=body.latency_ms,
        cost_usd=body.cost_usd,
        latency_budget_ms=body.latency_budget_ms,
        cost_budget_usd=body.cost_budget_usd,
        cost_floor_usd=body.cost_floor_usd,
        grounded_correctness_score=body.grounded_correctness_score,
        instruction_safety_score=body.instruction_safety_score,
        pairwise_preference_score=body.pairwise_preference_score,
    )


# -- Entry point -----------


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int_env("EIRETES_JUDGE_PORT", 8095, minimum=1, maximum=65535),
    )


if __name__ == "__main__":
    main()
