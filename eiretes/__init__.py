"""Eiretes — reference-based eval judge service for the EIREL subnet.

Public surface:
    EvalJudge          — LLM-as-judge client (eiretes/eval/judge.py)
    composite_score    — multiplicative composite (eiretes/eval/composite.py)
    EvalItem, EvalOutcome, EvalCompositeScore, EvalFeedbackDoc

The HTTP service (``eiretes.service``) exposes:
    GET  /healthz
    POST /v1/judge/eval              — judge call
    POST /v1/judge/eval/composite    — composite scorer

Per-miner feedback retrieval lives on owner-api directly
(hotkey-signed ``GET /v1/eval/feedback``); eiretes is purely the
judge service.
"""
from __future__ import annotations

from eiretes.eval.composite import composite_score
from eiretes.eval.judge import EvalJudge
from eiretes.eval.models import (
    EvalCompositeScore,
    EvalFeedbackDoc,
    EvalItem,
    EvalItemRecord,
    EvalOutcome,
    FailureMode,
    FeedbackPerItem,
    ItemKind,
    OracleSource,
    Outcome,
)

__all__ = [
    "EvalJudge",
    "composite_score",
    "EvalItem",
    "EvalOutcome",
    "EvalCompositeScore",
    "EvalItemRecord",
    "EvalFeedbackDoc",
    "FeedbackPerItem",
    "ItemKind",
    "OracleSource",
    "Outcome",
    "FailureMode",
]
