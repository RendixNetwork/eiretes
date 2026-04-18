from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from eiretes.models import JudgeResult
from eiretes.service import app


def _mock_judge_result() -> JudgeResult:
    return JudgeResult(
        model="test-model",
        rubric_name="test_rubric",
        score=0.85,
        rationale="Test rationale",
        latency_seconds=1.0,
        dimension_scores={"goal_fulfillment": 0.9},
        constraint_flags=[],
    )


def _mock_judge_client(**methods) -> AsyncMock:
    client = AsyncMock()
    client.aclose = AsyncMock(return_value=None)
    for name, ret in methods.items():
        getattr(client, name).return_value = ret
    return client


def test_healthz():
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


def test_catalog_endpoint_exposes_general_chat_only():
    mock_judge = _mock_judge_client()
    mock_judge.model = "mock-model"
    mock_judge.rubric_version = "general_chat_rubric_v1"
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.get("/v1/catalog")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rubric_version"] == "general_chat_rubric_v1"
    assert data["judge_model"] == "mock-model"
    families = data["families"]
    assert set(families) == {"general_chat"}
    entry = families["general_chat"]
    assert entry["rubric_name"] == "general_chat_quality_rubric_v1"
    assert entry["judge_mode"] == "judge_primary"
    assert entry["judge_weight"] == 1.0
    assert set(entry["dimensions"]) == {
        "goal_fulfillment",
        "correctness",
        "grounding",
        "conversation_coherence",
    }
    assert set(entry["supported_modes"]) == {"instant", "thinking"}
    # system_prompt and dimension_rubrics must NOT leak through the catalog endpoint
    for family in families.values():
        assert "system_prompt" not in family
        assert "system_prompt_by_mode" not in family
        assert "dimension_rubrics" not in family
        assert "active_system_prompt" not in family


def test_judge_endpoint():
    mock_judge = _mock_judge_client(judge=_mock_judge_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post("/v1/judge", json={
                "family_id": "general_chat",
                "prompt": "test prompt",
                "response_excerpt": "test response",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["score"] == 0.85
            assert data["model"] == "test-model"


def test_judge_endpoint_accepts_thinking_mode():
    mock_judge = _mock_judge_client(judge=_mock_judge_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge",
                json={
                    "family_id": "general_chat",
                    "prompt": "test prompt",
                    "response_excerpt": "test response",
                    "mode": "thinking",
                },
            )
    assert resp.status_code == 200
    mock_judge.judge.assert_called_once()
    kwargs = mock_judge.judge.call_args.kwargs
    assert kwargs["mode"] == "thinking"


def test_judge_endpoint_rejects_unknown_mode():
    mock_judge = _mock_judge_client(judge=_mock_judge_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge",
                json={
                    "family_id": "general_chat",
                    "prompt": "p",
                    "response_excerpt": "r",
                    "mode": "bogus",
                },
            )
    assert resp.status_code == 422


def test_judge_rejects_unknown_family_id():
    mock_judge = _mock_judge_client(judge=_mock_judge_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge",
                json={
                    "family_id": "analyst",
                    "prompt": "p",
                    "response_excerpt": "r",
                },
            )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "unknown_family_id"
    assert detail["family_id"] == "analyst"
    assert "general_chat" in detail["valid_families"]
    mock_judge.judge.assert_not_called()


def test_judge_request_enforces_prompt_length_limit():
    mock_judge = _mock_judge_client(judge=_mock_judge_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge",
                json={
                    "family_id": "general_chat",
                    "prompt": "x" * 40_000,
                    "response_excerpt": "r",
                },
            )
    assert resp.status_code == 422
