from __future__ import annotations

"""Behavior tests for LLMJudgeClient's deterministic fallback path.

These lock the fallback's invariants (bounded scores, expected dimension keys
for general_chat, monotonic response to signal presence) so future signal
tuning can't silently break the shape of JudgeResult that eirel-ai consumes.
"""

import pytest

from eiretes.judge.catalog import RUBRIC_CATALOG
from eiretes.judge.llm_judge import LLMJudgeClient


_GENERAL_CHAT_KEYS = {
    "goal_fulfillment",
    "correctness",
    "grounding",
    "conversation_coherence",
}


def _client() -> LLMJudgeClient:
    # No base_url/api_key → deterministic path only, no network.
    return LLMJudgeClient(model="det", rubric_version="test")


async def test_deterministic_bounded_zero_one():
    client = _client()
    result = await client.judge(
        family_id="general_chat",
        prompt="Evaluate the market.",
        response_excerpt=(
            "Evidence sources show mixed findings, however according to a study "
            "the recommendation is to invest. https://example.com"
        ),
    )
    assert 0.0 <= result.score <= 1.0
    for value in result.dimension_scores.values():
        assert 0.0 <= value <= 1.0


async def test_deterministic_empty_excerpt_returns_all_zero():
    client = _client()
    result = await client.judge(
        family_id="general_chat",
        prompt="Anything",
        response_excerpt="",
    )
    assert result.score == 0.0
    assert all(v == 0.0 for v in result.dimension_scores.values())


async def test_deterministic_general_chat_emits_all_four_dimensions():
    client = _client()
    result = await client.judge(
        family_id="general_chat",
        prompt="What is the current price?",
        response_excerpt=(
            "The price is $42 according to the source. https://example.com confirms it."
        ),
    )
    assert set(result.dimension_scores) == _GENERAL_CHAT_KEYS
    assert all(0.0 <= v <= 1.0 for v in result.dimension_scores.values())


async def test_deterministic_general_chat_emits_dimensions_in_thinking_mode():
    client = _client()
    response = " ".join(
        [
            "The price has fluctuated significantly over the last quarter."
        ]
        * 30
    )
    result = await client.judge(
        family_id="general_chat",
        prompt="Research the topic.",
        response_excerpt=response,
        mode="thinking",
    )
    assert set(result.dimension_scores) == _GENERAL_CHAT_KEYS
    assert result.metadata.get("mode") == "thinking"


async def test_deterministic_signal_presence_raises_correctness_score():
    """More citation/evidence signals should raise the correctness score."""
    client = _client()
    sparse = await client.judge(
        family_id="general_chat",
        prompt="Evaluate the market trends",
        response_excerpt="markets fluctuate sometimes",
    )
    rich = await client.judge(
        family_id="general_chat",
        prompt="Evaluate the market trends",
        response_excerpt=(
            "Markets fluctuate. According to a recent study the trends show "
            "evidence of growth. https://example.com confirms this finding."
        ),
    )
    assert rich.dimension_scores["correctness"] >= sparse.dimension_scores["correctness"]
    assert rich.dimension_scores["grounding"] >= sparse.dimension_scores["grounding"]


async def test_deterministic_metadata_flags_fallback():
    client = _client()
    result = await client.judge(
        family_id="general_chat",
        prompt="Review",
        response_excerpt="The recommended approach is to refactor the module.",
    )
    assert result.metadata.get("provider_used") is False
    # general_chat is the only family — judge_weight should be 1.0.
    assert result.metadata.get("judge_weight") == pytest.approx(
        float(RUBRIC_CATALOG["general_chat"]["judge_weight"])
    )


async def test_deterministic_rubric_variant_returns_single_dimension():
    client = _client()
    result = await client.judge(
        family_id="general_chat",
        prompt="Evaluate",
        response_excerpt="A careful answer with evidence and recommendation.",
        rubric_variant="goal_fulfillment",
    )
    assert set(result.dimension_scores) == {"goal_fulfillment"}


async def test_deterministic_preamble_lowers_coherence_score():
    client = _client()
    clean = await client.judge(
        family_id="general_chat",
        prompt="Tell me the answer to two plus two",
        response_excerpt="Two plus two equals four.",
    )
    preambled = await client.judge(
        family_id="general_chat",
        prompt="Tell me the answer to two plus two",
        response_excerpt="Certainly! I'd be happy to help. Two plus two equals four.",
    )
    assert (
        preambled.dimension_scores["conversation_coherence"]
        <= clean.dimension_scores["conversation_coherence"]
    )
