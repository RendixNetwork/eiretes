from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from eiretes.judge.catalog import RUBRIC_CATALOG
from eiretes.judge.llm_judge import LLMJudgeClient
from eiretes.models import JudgeResult


def _build_content(score: float, dimensions: dict[str, float] | None = None) -> str:
    dim_scores = dimensions or {"goal_fulfillment": score, "correctness": score}
    return json.dumps(
        {
            "overall_score": score,
            "dimension_scores": dim_scores,
            "constraint_flags": [],
            "rationale": f"Mock rationale at score {score}.",
        }
    )


def _completion(content: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _mock_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _make_single_score_transport(
    primary_score: float, ensemble_score: float | None
) -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        if "primary-judge" in str(request.url):
            return httpx.Response(200, json=_completion(_build_content(primary_score)))
        return httpx.Response(
            200,
            json=_completion(
                _build_content(
                    ensemble_score if ensemble_score is not None else primary_score
                )
            ),
        )

    return _mock_transport(_handler)


def _make_judge(
    *,
    primary_score: float = 0.8,
    ensemble_score: float | None = None,
    disagreement_threshold: float = 0.20,
) -> LLMJudgeClient:
    return LLMJudgeClient(
        model="test-model",
        rubric_version="v1",
        base_url="http://primary-judge",
        api_key="primary-key",
        ensemble_base_url="http://ensemble-judge" if ensemble_score is not None else "",
        ensemble_api_key="ensemble-key" if ensemble_score is not None else "",
        ensemble_disagreement_threshold=disagreement_threshold,
        transport=_make_single_score_transport(primary_score, ensemble_score),
    )


# ── catalog ───────────────────────────────────────────────────────────────────


def test_general_chat_rubric_has_ensemble_mode():
    assert RUBRIC_CATALOG["general_chat"].get("ensemble_mode") is True


# ── no ensemble configured → returns primary directly ─────────────────────────


async def test_judge_without_ensemble_returns_primary():
    client = LLMJudgeClient(
        model="test-model",
        rubric_version="v1",
        base_url="",
        api_key="",
        ensemble_base_url="",
        ensemble_api_key="",
    )
    result = await client.judge(
        family_id="general_chat",
        prompt="What are the risks?",
        response_excerpt="The main risks are A, B, and C.",
    )
    # Deterministic fallback — no ensemble
    assert isinstance(result, JudgeResult)
    assert "ensemble" not in (result.model or "")
    assert result.metadata.get("ensemble_used") is not True
    await client.aclose()


# ── ensemble averages primary and secondary scores ────────────────────────────


async def test_ensemble_averages_scores():
    client = _make_judge(primary_score=0.8, ensemble_score=0.6)
    result = await client.judge(
        family_id="general_chat",
        prompt="Research the topic.",
        response_excerpt="The answer covers all key points.",
    )
    assert result.score == pytest.approx(0.70, abs=0.01)
    assert result.metadata.get("primary_score") == pytest.approx(0.80, abs=0.01)
    assert result.metadata.get("secondary_score") == pytest.approx(0.60, abs=0.01)
    assert result.metadata.get("ensemble_used") is True
    assert "ensemble" in result.model
    await client.aclose()


async def test_ensemble_averages_dimension_scores():
    client = _make_judge(primary_score=0.8, ensemble_score=0.4)
    result = await client.judge(
        family_id="general_chat",
        prompt="Analyze the market.",
        response_excerpt="Market is driven by innovation.",
    )
    # Both primary and ensemble emit {goal_fulfillment, analytical_soundness}
    # so shared-key average of 0.8 and 0.4 is 0.6.
    assert result.dimension_scores.get("goal_fulfillment") == pytest.approx(0.6, abs=0.01)
    await client.aclose()


# ── disagreement flagging ────────────────────────────────────────────────────


async def test_disagreement_flag_added_when_above_threshold():
    client = _make_judge(
        primary_score=0.9, ensemble_score=0.5, disagreement_threshold=0.20
    )
    result = await client.judge(
        family_id="general_chat",
        prompt="Evaluate the study.",
        response_excerpt="The study is comprehensive.",
    )
    assert any("judge_disagreement" in flag for flag in result.constraint_flags)
    await client.aclose()


async def test_no_disagreement_flag_when_below_threshold():
    client = _make_judge(
        primary_score=0.80, ensemble_score=0.75, disagreement_threshold=0.20
    )
    result = await client.judge(
        family_id="general_chat",
        prompt="Assess the policy.",
        response_excerpt="The policy reduces risk.",
    )
    assert not any("judge_disagreement" in flag for flag in result.constraint_flags)
    await client.aclose()


# ── graceful fallback on ensemble failure ────────────────────────────────────


async def test_ensemble_failure_falls_back_to_primary():
    def _handler(request: httpx.Request) -> httpx.Response:
        if "failing-ensemble" in str(request.url):
            raise httpx.ConnectError("Connection refused")
        return httpx.Response(200, json=_completion(_build_content(0.75)))

    client = LLMJudgeClient(
        model="test-model",
        rubric_version="v1",
        base_url="http://primary",
        api_key="primary-key",
        ensemble_base_url="http://failing-ensemble",
        ensemble_api_key="ensemble-key",
        transport=_mock_transport(_handler),
    )
    result = await client.judge(
        family_id="general_chat",
        prompt="Topic analysis.",
        response_excerpt="A comprehensive answer.",
    )
    assert result.score == pytest.approx(0.75, abs=0.01)
    assert "ensemble" not in (result.model or "")
    await client.aclose()


# ── init params propagated correctly ─────────────────────────────────────────


def test_init_stores_ensemble_params():
    client = LLMJudgeClient(
        model="m",
        rubric_version="v1",
        ensemble_base_url="http://secondary.judge",
        ensemble_api_key="secret-key",
        ensemble_timeout_seconds=45.0,
        ensemble_disagreement_threshold=0.15,
    )
    assert client.ensemble_base_url == "http://secondary.judge"
    assert client.ensemble_api_key == "secret-key"
    assert client.ensemble_timeout_seconds == 45.0
    assert client.ensemble_disagreement_threshold == 0.15


def test_init_default_ensemble_empty_when_not_set(monkeypatch: pytest.MonkeyPatch):
    for env_var in (
        "EIREL_ENSEMBLE_JUDGE_BASE_URL",
        "EIREL_ENSEMBLE_JUDGE_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    client = LLMJudgeClient(model="m", rubric_version="v1")
    assert client.ensemble_base_url == ""
    assert client.ensemble_api_key == ""


# ── dimension key mismatch between primary and ensemble ─────────────────────


async def test_ensemble_dimension_key_mismatch_is_reported():
    def _handler(request: httpx.Request) -> httpx.Response:
        if "primary" in str(request.url):
            return httpx.Response(
                200,
                json=_completion(
                    _build_content(0.8, {"goal_fulfillment": 0.8, "correctness": 0.9})
                ),
            )
        return httpx.Response(
            200,
            json=_completion(
                _build_content(0.6, {"correctness": 0.5, "grounding": 0.7})
            ),
        )

    client = LLMJudgeClient(
        model="test-model",
        rubric_version="v1",
        base_url="http://primary",
        api_key="primary-key",
        ensemble_base_url="http://ensemble",
        ensemble_api_key="ensemble-key",
        transport=_mock_transport(_handler),
    )
    result = await client.judge(
        family_id="general_chat",
        prompt="Evaluate thoroughly.",
        response_excerpt="Considered analysis with reasoning.",
    )
    # correctness is the only shared dimension — (0.9 + 0.5) / 2 = 0.7
    assert result.dimension_scores["correctness"] == pytest.approx(0.7, abs=0.01)
    # primary-only dimension should carry through unchanged, not halved against 0.0
    assert result.dimension_scores["goal_fulfillment"] == pytest.approx(0.8, abs=0.01)
    # secondary-only dimension should also carry through unchanged
    assert result.dimension_scores["grounding"] == pytest.approx(0.7, abs=0.01)

    mismatch = result.metadata.get("dimension_coverage_mismatch") or {}
    assert "goal_fulfillment" in mismatch.get("primary_only", [])
    assert "grounding" in mismatch.get("secondary_only", [])
    assert "correctness" not in mismatch.get("primary_only", [])
    assert "correctness" not in mismatch.get("secondary_only", [])

    per_delta = result.metadata.get("per_dimension_delta") or {}
    assert per_delta.get("correctness") == pytest.approx(0.4, abs=0.01)
    await client.aclose()


async def test_ensemble_http_500_falls_back_to_primary(caplog: pytest.LogCaptureFixture):
    def _handler(request: httpx.Request) -> httpx.Response:
        if "broken-ensemble" in str(request.url):
            return httpx.Response(500, text="kaboom")
        return httpx.Response(200, json=_completion(_build_content(0.82)))

    client = LLMJudgeClient(
        model="test-model",
        rubric_version="v1",
        base_url="http://primary",
        api_key="primary-key",
        ensemble_base_url="http://broken-ensemble",
        ensemble_api_key="ensemble-key",
        transport=_mock_transport(_handler),
    )
    with caplog.at_level("WARNING", logger="eiretes.judge.llm_judge"):
        result = await client.judge(
            family_id="general_chat",
            prompt="Topic.",
            response_excerpt="A thoughtful response.",
        )

    assert result.score == pytest.approx(0.82, abs=0.01)
    assert "ensemble" not in (result.model or "")
    assert any("ensemble judge failed" in rec.message for rec in caplog.records)
    await client.aclose()


async def test_provider_invalid_schema_falls_back_to_deterministic(
    caplog: pytest.LogCaptureFixture,
):
    """C3: malformed dimension_scores should trigger deterministic fallback + warning."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "overall_score": 0.9,
                                    "dimension_scores": "not-a-dict",
                                    "rationale": "bad",
                                }
                            )
                        }
                    }
                ],
                "usage": {},
            },
        )

    client = LLMJudgeClient(
        model="test-model",
        rubric_version="v1",
        base_url="http://primary",
        api_key="primary-key",
        transport=_mock_transport(_handler),
    )
    with caplog.at_level("WARNING", logger="eiretes.judge.llm_judge"):
        result = await client.judge(
            family_id="general_chat",
            prompt="A prompt.",
            response_excerpt="A reasoned answer with because and therefore.",
        )

    assert result.metadata.get("provider_used") is False
    assert "provider_error" in result.metadata
    assert any("primary judge failed" in rec.message for rec in caplog.records)
    await client.aclose()
