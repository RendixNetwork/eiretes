from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from eiretes.models import AgreementJudgeResult
from eiretes.service import app


def _mock_result() -> AgreementJudgeResult:
    return AgreementJudgeResult(
        model="test-model",
        rubric_name="agreement_general_chat_v1:test",
        verdict="matches",
        agreement_score=1.0,
        rationale="Test rationale",
        latency_seconds=1.0,
        swap_applied=False,
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
    mock_judge.rubric_version = "agreement_general_chat_v1"
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.get("/v1/catalog")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rubric_version"] == "agreement_general_chat_v1"
    assert data["judge_model"] == "mock-model"
    families = data["families"]
    assert set(families) == {"general_chat"}
    entry = families["general_chat"]
    assert entry["rubric_name"] == "agreement_general_chat_v1"
    assert set(entry["verdicts"]) == {"matches", "partially_matches", "contradicts", "not_applicable"}
    for family in families.values():
        # system_prompt must never leak through catalog endpoint
        assert "system_prompt" not in family
        # process-metric fields are gone too
        assert "dimensions" not in family


def test_judge_agreement_endpoint():
    mock_judge = _mock_judge_client(judge_agreement=_mock_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge/agreement",
                json={
                    "family_id": "general_chat",
                    "prompt": "test prompt",
                    "response_a": "candidate answer",
                    "response_b": "baseline answer",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["verdict"] == "matches"
            assert data["agreement_score"] == 1.0
            assert data["model"] == "test-model"


def test_judge_agreement_endpoint_accepts_swap_and_task_context():
    mock_judge = _mock_judge_client(judge_agreement=_mock_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge/agreement",
                json={
                    "family_id": "general_chat",
                    "prompt": "p",
                    "response_a": "a",
                    "response_b": "b",
                    "swap": True,
                    "task_mode": "thinking",
                    "task_category": "factual_web",
                },
            )
    assert resp.status_code == 200
    mock_judge.judge_agreement.assert_called_once()
    kwargs = mock_judge.judge_agreement.call_args.kwargs
    assert kwargs["swap"] is True
    assert kwargs["task_mode"] == "thinking"
    assert kwargs["task_category"] == "factual_web"


def test_judge_agreement_rejects_unknown_family_id():
    mock_judge = _mock_judge_client(judge_agreement=_mock_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge/agreement",
                json={
                    "family_id": "analyst",
                    "prompt": "p",
                    "response_a": "a",
                    "response_b": "b",
                },
            )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "unknown_family_id"
    mock_judge.judge_agreement.assert_not_called()


def test_judge_agreement_enforces_prompt_length_limit():
    mock_judge = _mock_judge_client(judge_agreement=_mock_result())
    with patch("eiretes.service._build_judge", return_value=mock_judge):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/judge/agreement",
                json={
                    "family_id": "general_chat",
                    "prompt": "x" * 40_000,
                    "response_a": "a",
                    "response_b": "b",
                },
            )
    assert resp.status_code == 422
