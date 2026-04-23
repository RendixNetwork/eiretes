from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgreementVerdict = Literal["matches", "partially_matches", "contradicts", "not_applicable"]


# Verdict → agreement score mapping. Used by both the judge (to derive the
# scalar from the verdict) and by downstream aggregation (to confirm the
# conversion). Preserved as a single source of truth.
VERDICT_SCORES: dict[str, float] = {
    "matches": 1.0,
    "partially_matches": 0.6,
    "not_applicable": 0.7,
    "contradicts": 0.0,
}


class AgreementJudgeResult(BaseModel):
    """Outcome-only judge result returned to callers.

    The judge compares a candidate agent's final answer against the OpenAI
    baseline reference answer. No process metrics (citations, dimensions,
    style) participate in scoring.
    """

    model: str
    rubric_name: str
    verdict: AgreementVerdict
    agreement_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    latency_seconds: float = Field(ge=0.0)
    swap_applied: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderAgreementResponse(BaseModel):
    """Strict validation of the JSON body returned by an upstream LLM."""

    verdict: AgreementVerdict
    rationale: str = ""
