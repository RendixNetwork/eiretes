from __future__ import annotations

"""Shape and behavior tests for the general_chat rubric.

Locks the catalog structure (dimension keys, weight sum, mode-specific system
prompts) and the deterministic judge's dimension key set so downstream
consumers in eirel-ai can rely on a stable contract.
"""

import pytest

from eiretes.judge.catalog import RUBRIC_CATALOG, resolve_rubric_spec
from eiretes.judge.llm_judge import LLMJudgeClient
from eiretes.judge.rubrics.general_chat import GENERAL_CHAT_QUALITY_RUBRIC

_EXPECTED_DIMENSIONS = {
    "goal_fulfillment",
    "correctness",
    "grounding",
    "conversation_coherence",
}


def test_catalog_has_only_general_chat_family():
    assert set(RUBRIC_CATALOG) == {"general_chat"}


def test_general_chat_rubric_shape():
    rubric = RUBRIC_CATALOG["general_chat"]
    assert rubric["rubric_name"] == "general_chat_quality_rubric_v1"
    assert rubric["judge_weight"] == 1.0
    assert rubric["judge_mode"] == "judge_primary"
    assert rubric["ensemble_mode"] is True
    assert set(rubric["dimensions"]) == _EXPECTED_DIMENSIONS


def test_general_chat_weights_sum_to_one():
    weights = GENERAL_CHAT_QUALITY_RUBRIC["weights"]
    assert set(weights) == _EXPECTED_DIMENSIONS
    assert sum(weights.values()) == pytest.approx(1.0)


def test_general_chat_system_prompts_for_both_modes():
    prompts = GENERAL_CHAT_QUALITY_RUBRIC["system_prompt_by_mode"]
    assert set(prompts) == {"instant", "thinking"}
    assert all(isinstance(value, str) and value.strip() for value in prompts.values())
    assert prompts["instant"] != prompts["thinking"]


def test_general_chat_dimension_rubrics_have_five_anchors_each():
    for dim, anchors in GENERAL_CHAT_QUALITY_RUBRIC["dimension_rubrics"].items():
        for level in ("score_5", "score_4", "score_3", "score_2", "score_1"):
            assert anchors.get(level), f"{dim} missing {level}"


def test_resolve_rubric_spec_instant_mode():
    spec = resolve_rubric_spec("general_chat", mode="instant")
    assert spec["rubric_name"] == "general_chat_quality_rubric_v1"
    assert spec["active_mode"] == "instant"
    assert spec["active_system_prompt"] == GENERAL_CHAT_QUALITY_RUBRIC[
        "system_prompt_by_mode"
    ]["instant"]


def test_resolve_rubric_spec_thinking_mode():
    spec = resolve_rubric_spec("general_chat", mode="thinking")
    assert spec["active_mode"] == "thinking"
    assert spec["active_system_prompt"] == GENERAL_CHAT_QUALITY_RUBRIC[
        "system_prompt_by_mode"
    ]["thinking"]


def test_resolve_rubric_spec_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family_id"):
        resolve_rubric_spec("analyst", mode="instant")


def test_resolve_rubric_spec_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_rubric_spec("general_chat", mode="bogus")


def test_resolve_rubric_spec_returns_shallow_copy():
    spec_a = resolve_rubric_spec("general_chat", mode="instant")
    spec_a["active_mode"] = "tampered"
    spec_b = resolve_rubric_spec("general_chat", mode="instant")
    assert spec_b["active_mode"] == "instant"


async def test_llm_judge_dimension_scores_returns_four_keys():
    client = LLMJudgeClient(model="det", rubric_version="test")
    result = await client.judge(
        family_id="general_chat",
        prompt="What is the capital of France?",
        response_excerpt=(
            "The capital of France is Paris. According to evidence from "
            "https://example.com this is well-established."
        ),
    )
    assert set(result.dimension_scores) == _EXPECTED_DIMENSIONS
    assert result.metadata.get("mode") == "instant"


async def test_llm_judge_thinking_mode_metadata():
    client = LLMJudgeClient(model="det", rubric_version="test")
    result = await client.judge(
        family_id="general_chat",
        prompt="Explain in detail",
        response_excerpt="A long thoughtful response. " * 30,
        mode="thinking",
    )
    assert result.metadata.get("mode") == "thinking"
    assert set(result.dimension_scores) == _EXPECTED_DIMENSIONS
