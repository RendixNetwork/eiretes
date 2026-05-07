"""Per-miner feedback never reveals expected_answer; cross-hotkey access denied."""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from eiretes.eval.feedback import FeedbackStore
from eiretes.eval.models import (
    EvalCompositeScore,
    EvalItemRecord,
    EvalOutcome,
)


def _record(
    *,
    template_id: str = "t-1",
    outcome_name: str = "correct",
    failure_mode: str | None = None,
    candidate_response: str = "ok",
    expected_answer: str = "the secret answer",
    composite: float = 1.0,
) -> EvalItemRecord:
    return EvalItemRecord(
        item_id=uuid4().hex[:8],
        kind="single_turn_factual",
        template_id=template_id,
        composite=EvalCompositeScore(
            composite=composite,
            outcome_score=1.0 if outcome_name == "correct" else 0.5,
            tool_attestation_factor=1.0,
            efficiency_factor=1.0,
            hallucination_knockout=1.0,
            cost_attestation_knockout=1.0,
        ),
        outcome=EvalOutcome(
            outcome=outcome_name,  # type: ignore[arg-type]
            failure_mode=failure_mode,  # type: ignore[arg-type]
            guidance="ground answers in the document",
        ),
        candidate_response=candidate_response,
        expected_answer=expected_answer,
    )


def test_feedback_doc_aggregates_failure_modes():
    store = FeedbackStore()
    records = [
        _record(outcome_name="wrong", failure_mode="wrong_fact", composite=0.0),
        _record(outcome_name="wrong", failure_mode="wrong_fact", composite=0.0),
        _record(outcome_name="partial", failure_mode="incomplete", composite=0.5),
        _record(outcome_name="correct", composite=1.0),
    ]
    doc = store.write_for_records(
        run_id="run-1", miner_hotkey="hk1", records=records,
    )
    assert doc.n_items == 4
    # 2 + 1 = 3/4 → composite mean = (0+0+0.5+1.0)/4 = 0.375
    assert doc.composite_score == pytest.approx(0.375)
    assert doc.per_failure_mode_counts == {
        "wrong_fact": 2,
        "incomplete": 1,
    }
    assert "wrong_fact" in doc.largest_gap


def test_feedback_doc_does_not_leak_expected_answer():
    """The feedback payload (sent to the miner) must NEVER contain the
    expected_answer verbatim."""
    store = FeedbackStore()
    records = [
        _record(
            expected_answer="THE_SECRET_ANSWER_TOKEN_xyz",
            candidate_response="my response",
        ),
    ]
    doc = store.write_for_records(
        run_id="run-2", miner_hotkey="hk2", records=records,
    )
    payload_json = doc.model_dump_json()
    assert "THE_SECRET_ANSWER_TOKEN_xyz" not in payload_json


def test_feedback_store_isolates_by_hotkey():
    store = FeedbackStore()
    store.write_for_records(
        run_id="r", miner_hotkey="alice", records=[_record()],
    )
    store.write_for_records(
        run_id="r", miner_hotkey="bob", records=[_record(composite=0.0, outcome_name="wrong")],
    )
    assert store.get(run_id="r", miner_hotkey="alice").composite_score == 1.0
    assert store.get(run_id="r", miner_hotkey="bob").composite_score == 0.0
    assert store.get(run_id="r", miner_hotkey="charlie") is None
