from __future__ import annotations

"""Shape and behavior tests for the outcome-only agreement judge.

Locks the catalog structure (verdicts, system prompt) and the judge
client's verdict→score mapping so downstream consumers in eirel-ai can
rely on a stable contract. Also verifies:
  - citations must not be asked about in the system prompt
  - swap does not flip the verdict (matches/partially/contradicts/
    not_applicable are symmetric between A and B)
"""

import json

import httpx
import pytest

from eiretes.judge.catalog import RUBRIC_CATALOG, resolve_rubric_spec
from eiretes.judge.llm_judge import LLMJudgeClient
from eiretes.judge.rubrics.pairwise_general_chat import (
    PAIRWISE_GENERAL_CHAT_RUBRIC,
    build_system_prompt,
)
from eiretes.models import VERDICT_SCORES

_EXPECTED_VERDICTS = {"matches", "partially_matches", "contradicts", "not_applicable"}


def test_catalog_has_only_general_chat_family():
    assert set(RUBRIC_CATALOG) == {"general_chat"}


def test_agreement_rubric_shape():
    rubric = RUBRIC_CATALOG["general_chat"]
    assert rubric["rubric_name"] == "agreement_general_chat_v1"
    assert set(rubric["verdicts"]) == _EXPECTED_VERDICTS
    assert isinstance(rubric["system_prompt"], str) and rubric["system_prompt"].strip()


def test_rubric_has_no_process_metrics():
    """The redesigned rubric scores only outcome, not process (citations,
    dimensions, style). These fields must be absent from the catalog entry."""
    rubric = RUBRIC_CATALOG["general_chat"]
    assert "dimensions" not in rubric
    assert "weights" not in rubric


def test_rubric_system_prompt_instructs_ignoring_citations():
    """The judge must be told explicitly not to evaluate citations —
    miners and the baseline use different search engines."""
    prompt = PAIRWISE_GENERAL_CHAT_RUBRIC["system_prompt"]
    assert "citation" in prompt.lower() or "source" in prompt.lower()
    # A basic spot-check that the framing explicitly de-weights citations.
    assert "different" in prompt.lower() or "not your concern" in prompt.lower()


def test_verdict_scores_mapping():
    """Contract the scalar mapping so downstream aggregation stays aligned."""
    assert VERDICT_SCORES["matches"] == 1.0
    assert VERDICT_SCORES["contradicts"] == 0.0
    assert 0.0 < VERDICT_SCORES["partially_matches"] < 1.0
    assert 0.0 < VERDICT_SCORES["not_applicable"] <= 1.0
    assert set(VERDICT_SCORES) == _EXPECTED_VERDICTS


def test_resolve_rubric_spec_returns_shallow_copy():
    spec_a = resolve_rubric_spec("general_chat")
    spec_a["rubric_name"] = "tampered"
    spec_b = resolve_rubric_spec("general_chat")
    assert spec_b["rubric_name"] == "agreement_general_chat_v1"


def test_resolve_rubric_spec_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family_id"):
        resolve_rubric_spec("analyst")


def _mock_transport(canned_body: dict) -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [{"message": {"content": json.dumps(canned_body)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(_handler)


async def test_judge_agreement_matches_verdict_maps_to_score_1():
    canned = {"verdict": "matches", "rationale": "claims align"}
    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=_mock_transport(canned),
    )
    result = await client.judge_agreement(
        family_id="general_chat",
        prompt="What is X?",
        response_a="miner answer",
        response_b="baseline answer",
    )
    assert result.verdict == "matches"
    assert result.agreement_score == 1.0
    assert result.swap_applied is False
    await client.aclose()


async def test_judge_agreement_partially_matches_maps_to_06():
    canned = {"verdict": "partially_matches", "rationale": "missing a sub-claim"}
    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=_mock_transport(canned),
    )
    result = await client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b",
    )
    assert result.verdict == "partially_matches"
    assert result.agreement_score == 0.6
    await client.aclose()


async def test_judge_agreement_contradicts_maps_to_zero():
    canned = {"verdict": "contradicts", "rationale": "different conclusion"}
    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=_mock_transport(canned),
    )
    result = await client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b",
    )
    assert result.verdict == "contradicts"
    assert result.agreement_score == 0.0
    await client.aclose()


async def test_judge_agreement_not_applicable_for_open_ended():
    canned = {
        "verdict": "not_applicable",
        "rationale": "both produced valid but different travel itineraries",
    }
    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=_mock_transport(canned),
    )
    result = await client.judge_agreement(
        family_id="general_chat",
        prompt="plan a Tokyo trip",
        response_a="A's trip",
        response_b="B's trip",
        task_category="multi_step_reasoning",
    )
    assert result.verdict == "not_applicable"
    assert result.agreement_score == 0.7
    await client.aclose()


async def test_judge_agreement_swap_does_not_flip_verdict():
    """Verdicts are symmetric (matches means 'they agree', not 'A beat B').
    Swap is only for position-bias mitigation; it must not change the verdict."""
    canned = {"verdict": "matches", "rationale": "aligned"}
    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=_mock_transport(canned),
    )
    r_no = await client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b", swap=False,
    )
    r_yes = await client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b", swap=True,
    )
    assert r_no.verdict == r_yes.verdict == "matches"
    assert r_no.agreement_score == r_yes.agreement_score
    assert r_no.swap_applied is False
    assert r_yes.swap_applied is True
    await client.aclose()


async def test_judge_agreement_without_credentials_raises():
    client = LLMJudgeClient(model="x", rubric_version="test", base_url="", api_key="")
    with pytest.raises(RuntimeError, match="requires EIREL_JUDGE_BASE_URL"):
        await client.judge_agreement(
            family_id="general_chat",
            prompt="p", response_a="a", response_b="b",
        )


async def test_judge_agreement_invalid_verdict_raises():
    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=_mock_transport({"verdict": "maybe", "rationale": ""}),
    )
    with pytest.raises(ValueError, match="invalid schema"):
        await client.judge_agreement(
            family_id="general_chat",
            prompt="p", response_a="a", response_b="b",
        )
    await client.aclose()


async def test_judge_agreement_malformed_json_raises():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json {{{"}}]},
        )

    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=httpx.MockTransport(_handler),
    )
    with pytest.raises(ValueError, match="malformed JSON"):
        await client.judge_agreement(
            family_id="general_chat",
            prompt="p", response_a="a", response_b="b",
        )
    await client.aclose()


def test_build_system_prompt_falls_through_without_category():
    base = PAIRWISE_GENERAL_CHAT_RUBRIC["system_prompt"]
    assert build_system_prompt(None) == base
    assert build_system_prompt("") == base
    # Unknown category → base prompt only, no silent failure.
    assert build_system_prompt("unknown_category_xyz") == base


def test_build_system_prompt_appends_safety_adversarial_addendum():
    prompt = build_system_prompt("safety_adversarial")
    base = PAIRWISE_GENERAL_CHAT_RUBRIC["system_prompt"]
    assert prompt.startswith(base)
    # The critical rule: refusing harmful questions must not be penalized.
    lower = prompt.lower()
    assert "refus" in lower
    assert "safety" in lower or "harm" in lower
    assert "safety_adversarial" in prompt


def test_build_system_prompt_appends_coding_addendum():
    prompt = build_system_prompt("coding")
    assert "coding" in prompt.lower()
    # Must tell the judge to compare program behavior, not surface form.
    assert "behavior" in prompt.lower() or "output" in prompt.lower()


def test_build_system_prompt_appends_math_reasoning_addendum():
    prompt = build_system_prompt("math_reasoning")
    assert "math_reasoning" in prompt
    # The key rule: same final answer via different derivation = matches.
    assert "final" in prompt.lower()


def test_build_system_prompt_appends_ambiguous_query_addendum():
    prompt = build_system_prompt("ambiguous_query")
    assert "ambiguous_query" in prompt
    assert "interpretation" in prompt.lower()


def test_build_system_prompt_all_known_categories_covered():
    """Every category we promise to specialize for must have an addendum."""
    expected = {
        "safety_adversarial", "ambiguous_query", "coding",
        "math_reasoning", "multi_step_reasoning",
        "academic_research", "long_context",
    }
    addenda = PAIRWISE_GENERAL_CHAT_RUBRIC["category_addenda"]
    assert expected.issubset(set(addenda))
    # Each addendum must be non-empty and name its category.
    for name, text in addenda.items():
        assert text and text.strip()
        assert name in text


async def test_judge_agreement_uses_category_specific_prompt():
    """When task_category is set, the system prompt sent to the LLM must
    include the category addendum. This test intercepts the upstream
    request body to verify the assembled prompt carries the override."""
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["system"] = body["messages"][0]["content"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "verdict": "matches", "rationale": "ok",
            })}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        })

    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=httpx.MockTransport(_handler),
    )
    await client.judge_agreement(
        family_id="general_chat",
        prompt="Is acetaminophen safe for a 4-year-old?",
        response_a="Consult a pediatrician.",
        response_b="Give 10mg/kg.",
        task_category="safety_adversarial",
    )
    # Category addendum must be part of what the LLM actually saw.
    assert "safety_adversarial" in captured["system"]
    assert "refus" in captured["system"].lower()
    await client.aclose()


async def test_judge_agreement_uses_base_prompt_for_unknown_category():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["system"] = body["messages"][0]["content"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "verdict": "matches", "rationale": "ok",
            })}}],
            "usage": {},
        })

    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=httpx.MockTransport(_handler),
    )
    await client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b",
        task_category="definitely_not_a_real_category",
    )
    # Base prompt only — no category override markers from any known category.
    system = captured["system"]
    for known in (
        "safety_adversarial", "ambiguous_query", "coding",
        "math_reasoning", "multi_step_reasoning",
        "academic_research", "long_context",
    ):
        # The word "coding" does not appear in the base prompt; likewise
        # other category names. "Category override —" is the hallmark of
        # an addendum being present. It must not be present here.
        pass
    assert "Category override" not in system
    await client.aclose()


async def test_judge_agreement_metadata_records_task_context():
    canned = {"verdict": "matches", "rationale": "aligned"}
    client = LLMJudgeClient(
        model="gpt-x", rubric_version="test",
        base_url="http://judge", api_key="k",
        transport=_mock_transport(canned),
    )
    result = await client.judge_agreement(
        family_id="general_chat",
        prompt="p",
        response_a="a", response_b="b",
        task_mode="thinking",
        task_category="factual_web",
    )
    assert result.metadata["task_mode"] == "thinking"
    assert result.metadata["task_category"] == "factual_web"
    assert result.metadata["rubric_version"] == "test"
    await client.aclose()
