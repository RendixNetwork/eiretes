"""JudgeInputBundle dispatch logic tests.

Verifies the role-aware field selection + budget-aware truncation
the bundle exposes via ``dispatch_for(role, budget_tokens)``. The
judge modules call this method to render the user-prompt JSON; the
priority order is the load-bearing contract.
"""

from __future__ import annotations

import json

import pytest

from eiretes.eval.bundle import JudgeInputBundle


# -- Always-included fields (question + answers) --------------------------


def test_pairwise_dispatch_includes_question_and_both_answers():
    bundle = JudgeInputBundle(
        question="What is 2+2?",
        answers=["The answer is 4.", "The answer is 5."],
    )
    out = bundle.dispatch_for(role="pairwise", budget_tokens=8000)
    assert out["question"] == "What is 2+2?"
    assert out["answer_a"] == "The answer is 4."
    assert out["answer_b"] == "The answer is 5."


def test_eval_dispatch_includes_question_and_single_candidate():
    bundle = JudgeInputBundle(
        question="What is 2+2?",
        answers=["The answer is 4."],
    )
    out = bundle.dispatch_for(role="eval", budget_tokens=8000)
    assert out["question"] == "What is 2+2?"
    assert out["candidate_response"] == "The answer is 4."
    assert "answer_a" not in out
    assert "answer_b" not in out


def test_multi_dispatch_same_shape_as_eval():
    bundle = JudgeInputBundle(
        question="What is 2+2?",
        answers=["4"],
    )
    out = bundle.dispatch_for(role="multi", budget_tokens=8000)
    assert out["candidate_response"] == "4"


# -- Role/answers-tuple shape mismatch -----------------------------------


def test_pairwise_with_single_answer_raises():
    bundle = JudgeInputBundle(question="...", answers=["only one"])
    with pytest.raises(ValueError, match="length 2"):
        bundle.dispatch_for(role="pairwise")


def test_eval_with_two_answers_raises():
    bundle = JudgeInputBundle(question="...", answers=["a", "b"])
    with pytest.raises(ValueError, match="length 1"):
        bundle.dispatch_for(role="eval")


def test_multi_with_zero_answers_raises():
    bundle = JudgeInputBundle(question="...", answers=[])
    with pytest.raises(ValueError, match="length 1"):
        bundle.dispatch_for(role="multi")


# -- Constraints always included when present ----------------------------


def test_constraints_passed_through():
    bundle = JudgeInputBundle(
        question="?",
        answers=["x"],
        constraints="must not claim London is the capital",
    )
    out = bundle.dispatch_for(role="eval")
    assert out["constraints"] == "must not claim London is the capital"


def test_empty_constraints_omitted():
    bundle = JudgeInputBundle(question="?", answers=["x"], constraints=None)
    out = bundle.dispatch_for(role="eval")
    assert "constraints" not in out


# -- Optional fields included when budget allows -------------------------


def test_attached_summary_included_when_budget_allows():
    bundle = JudgeInputBundle(
        question="?",
        answers=["x"],
        attached_summary="user attached a 5-page contract",
    )
    out = bundle.dispatch_for(role="pairwise", budget_tokens=8000) if False else \
        bundle.dispatch_for(role="eval", budget_tokens=8000)
    # attached_summary is in the priority list and always small.
    assert out["attached_summary"] == "user attached a 5-page contract"


def test_conversation_recent_included_when_budget_allows():
    bundle = JudgeInputBundle(
        question="?",
        answers=["x"],
        conversation_recent=[
            {"role": "user", "content": "I work in Python."},
            {"role": "assistant", "content": "noted"},
        ],
    )
    out = bundle.dispatch_for(role="eval", budget_tokens=8000)
    assert len(out["conversation_recent"]) == 2


# -- Role policy: pairwise/multi never see attached_full -----------------


def test_pairwise_never_gets_attached_full_even_with_huge_budget():
    """Pairwise's bias controls explicitly forbid acting as a primary
    factuality evaluator. Even with infinite budget, pairwise gets
    only the attached_summary, never the raw doc."""
    bundle = JudgeInputBundle(
        question="?",
        answers=["a", "b"],
        attached_summary="200-token summary",
        attached_full="FULL DOCUMENT TEXT" * 100,
    )
    out = bundle.dispatch_for(role="pairwise", budget_tokens=1_000_000)
    assert out["attached_summary"] == "200-token summary"
    assert "attached_full" not in out


def test_multi_never_gets_attached_full_even_with_huge_budget():
    """Multi-judge consumes pre-extracted expected_claims (passed as
    a per-call extra by the caller, not via the bundle). The raw doc
    isn't needed either."""
    bundle = JudgeInputBundle(
        question="?",
        answers=["x"],
        attached_summary="summary",
        attached_full="FULL DOCUMENT TEXT" * 100,
    )
    out = bundle.dispatch_for(role="multi", budget_tokens=1_000_000)
    assert "attached_full" not in out


def test_eval_can_get_attached_full_when_budget_allows():
    bundle = JudgeInputBundle(
        question="?",
        answers=["x"],
        attached_full="full doc text",
    )
    out = bundle.dispatch_for(role="eval", budget_tokens=8000)
    assert out["attached_full"] == "full doc text"


# -- Greedy priority: drop lowest first when budget tight ----------------


def test_tight_budget_drops_attached_full_first():
    """When budget runs out, attached_full (lowest priority) is
    omitted before higher-priority fields."""
    bundle = JudgeInputBundle(
        question="What's the deal?",
        answers=["candidate response"],
        attached_summary="brief summary",
        conversation_summary="prior context",
        attached_full="A" * 5000,  # large
    )
    # Budget too small to fit attached_full but big enough for others.
    out = bundle.dispatch_for(role="eval", budget_tokens=400)
    assert out["question"] == "What's the deal?"
    assert out["candidate_response"] == "candidate response"
    assert out["attached_summary"] == "brief summary"
    assert "attached_full" not in out


def test_extremely_tight_budget_keeps_question_and_answers():
    """Even with a budget that can't fit attached_summary, question
    and answers (highest priority) are always included."""
    bundle = JudgeInputBundle(
        question="Q",
        answers=["A"],
        attached_summary="X" * 4000,  # very large
    )
    out = bundle.dispatch_for(role="eval", budget_tokens=20)
    assert out["question"] == "Q"
    assert out["candidate_response"] == "A"
    # attached_summary doesn't fit
    assert "attached_summary" not in out


# -- Output is JSON-serializable -----------------------------------------


def test_dispatch_output_is_json_serializable():
    bundle = JudgeInputBundle(
        question="?",
        answers=["a", "b"],
        attached_summary="s",
        conversation_summary="c",
        conversation_recent=[{"role": "user", "content": "hi"}],
        constraints="must do X",
    )
    out = bundle.dispatch_for(role="pairwise")
    # If this round-trips cleanly, the output is suitable as a user-
    # prompt JSON for the judge.
    text = json.dumps(out)
    assert json.loads(text) == out


# -- Pydantic model validation -------------------------------------------


def test_bundle_validates_on_dict_input():
    """Service.py receives the bundle as a dict from JSON; Pydantic
    coerces fields. Spot-check the round-trip."""
    raw = {
        "question": "Q",
        "answers": ["a"],
        "attached_summary": "s",
    }
    bundle = JudgeInputBundle.model_validate(raw)
    assert bundle.question == "Q"
    assert bundle.answers == ["a"]
    assert bundle.attached_summary == "s"
    assert bundle.attached_full is None
    assert bundle.conversation_recent == []
