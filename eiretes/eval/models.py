"""Eval pipeline data models.

Plain Pydantic models for the simple, hidden, server-attested eval. The
public API surface is small on purpose: ``EvalReport`` is everything
external callers see; ``EvalCompositeScore``/``EvalOutcome`` are
internal building blocks.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# -- Item kind / oracle source --------------------------------------------


# Mirrors eirel-eval-pool's pool kinds. Free-form so eiretes doesn't have
# to upgrade in lockstep when the pool adds a kind — judges treat this as
# informational metadata only.
ItemKind = str

# Where ``expected_claims`` came from. ``three_oracle`` items are subject
# to ``disputed`` outcomes when the candidate's answer is plausibly
# correct but disagrees with the validator-side reconciler's consensus
# claims. ``deterministic`` items (live_endpoint, sandbox_python, span F1,
# regex graders) have no such uncertainty — the gold is computed at pool
# render time.
OracleSource = Literal[
    "three_oracle",
    "deterministic",
]


# -- Item --------------------------------------------------------------------


class EvalItem(BaseModel):
    """One rendered eval item — exists only for the duration of one run.

    Never persisted to the repo; never returned to the candidate.
    """

    item_id: str = Field(min_length=1)
    kind: ItemKind
    prompt: str | None = None
    turns: list[dict[str, str]] | None = None
    expected_answer: str
    must_not_claim: list[str] = Field(default_factory=list)
    required_tool: str | None = None
    oracle_source: OracleSource
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    template_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# -- Per-item judge outcome ------------------------------------------------


Outcome = Literal[
    "correct",
    "partial",
    "wrong",
    "hallucinated",
    "refused",
    "disputed",
]

FailureMode = Literal[
    "missing_tool_use",
    "wrong_fact",
    "missing_grounding",
    "off_topic",
    "incomplete",
    "over_refusal",
    "hallucinated_claim",
]


class EvalOutcome(BaseModel):
    outcome: Outcome
    failure_mode: FailureMode | None = None
    guidance: str = ""


# -- Composite -------------------------------------------------------------


class EvalCompositeScore(BaseModel):
    """Per-item composite. Multiplicative formula closes sandbagging.

    Outer-dimension gates (``grounded_correctness_gate``,
    ``instruction_safety_gate``) zero the score when factual
    correctness or safety fails — a polished hallucinator can't
    win on style. Pairwise comparison adds a small ±0.10 bonus on
    top so equally-correct miners are still rank-ordered against the
    OpenAI baseline.
    """

    composite: float = Field(ge=0.0, le=1.0)
    outcome_score: float = Field(ge=0.0, le=1.0)
    tool_attestation_factor: float = Field(ge=0.0, le=1.0)
    efficiency_factor: float = Field(ge=0.0, le=1.0)
    hallucination_knockout: float = Field(ge=0.0, le=1.0)
    cost_attestation_knockout: float = Field(ge=0.0, le=1.0)
    grounded_correctness_gate: float = Field(default=1.0, ge=0.0, le=1.0)
    instruction_safety_gate: float = Field(default=1.0, ge=0.0, le=1.0)
    # Server-attested response-safety knockout — regex denylist +
    # chat-template token leakage. Zero on a hit; gameproof relative
    # to the LLM-judged ``instruction_safety_gate`` (which is still
    # in the formula as a soft signal).
    safety_attestation_knockout: float = Field(default=1.0, ge=0.0, le=1.0)
    pairwise_bonus: float = Field(default=0.0, ge=-0.10, le=0.10)
    knockout_reason: str | None = None


class EvalItemRecord(BaseModel):
    """Internal — joins item + outcome + composite + dispatch info.

    Never publicly broadcast; surfaces only in the per-miner feedback
    document and in admin breakdowns.
    """

    item_id: str
    kind: ItemKind
    template_id: str
    composite: EvalCompositeScore
    outcome: EvalOutcome
    candidate_response: str
    expected_answer: str
    latency_ms: int = 0
    cost_usd: float = 0.0
    tool_calls_observed: list[str] = Field(default_factory=list)
    error: str | None = None


# -- Public report --------------------------------------------------------


class EvalReport(BaseModel):
    """The only thing external callers see for a run.

    Per-item details live in :class:`EvalFeedbackDoc` and are gated on
    hotkey ownership.
    """

    run_id: str
    candidate_label: str
    n_items: int
    n_completed: int
    mean_composite: float = Field(ge=0.0, le=1.0)
    started_at: datetime
    finished_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


# -- Per-miner feedback ---------------------------------------------------


class FeedbackPerItem(BaseModel):
    """One item slice in a miner's feedback. Categorical guidance only —
    never reveals ``expected_answer`` verbatim.
    """

    item_kind: ItemKind
    prompt_excerpt: str
    your_response_excerpt: str
    outcome: Outcome
    failure_mode: FailureMode | None = None
    guidance: str = ""
    composite: float = Field(ge=0.0, le=1.0)


class EvalFeedbackDoc(BaseModel):
    run_id: str
    miner_hotkey: str
    composite_score: float = Field(ge=0.0, le=1.0)
    n_items: int
    per_failure_mode_counts: dict[str, int] = Field(default_factory=dict)
    largest_gap: str = ""
    per_item: list[FeedbackPerItem] = Field(default_factory=list)
    created_at: datetime


__all__ = [
    "EvalItem",
    "EvalOutcome",
    "EvalCompositeScore",
    "EvalItemRecord",
    "EvalReport",
    "EvalFeedbackDoc",
    "FeedbackPerItem",
    "ItemKind",
    "OracleSource",
    "Outcome",
    "FailureMode",
]
