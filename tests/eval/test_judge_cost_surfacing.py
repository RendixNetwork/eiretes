"""Verify each judge endpoint surfaces ``cost_usd`` from the upstream
``usage`` block on its HTTP response.

The validator engine in eirel-ai reads ``cost_usd`` from these
responses to populate ``TaskMinerResult.judge_cost_usd``. Without
this, the per-validator cost dashboard shows $0 even on successful
judgments.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from eiretes.service import app


def _scripted_response(content: dict, *, prompt_tokens: int, completion_tokens: int):
    """Build a Chutes-shaped chat-completions response with content + usage."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            },
        )

    return _handler


def _patch_async_client(monkeypatch, module_name: str, handler):
    """Swap the module's ``httpx.AsyncClient`` for one bound to ``handler``."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    class _StubAsyncClient(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(f"{module_name}.httpx.AsyncClient", _StubAsyncClient)


def _set_judge_env(monkeypatch):
    monkeypatch.setenv("EIREL_EVAL_JUDGE_BASE_URL", "https://llm.chutes.ai/v1")
    monkeypatch.setenv("EIREL_EVAL_JUDGE_API_KEY", "tok")
    monkeypatch.setenv("EIREL_EVAL_JUDGE_MODEL", "zai-org/GLM-5.1-TEE")


# -- /v1/judge/pairwise -------------------------------------------------


def test_pairwise_response_includes_cost_usd(monkeypatch):
    _set_judge_env(monkeypatch)
    handler = _scripted_response(
        {"winner": "A", "confidence": 0.8, "reason": "ok"},
        prompt_tokens=1000, completion_tokens=500,
    )
    _patch_async_client(monkeypatch, "eiretes.eval.pairwise", handler)

    client = TestClient(app)
    resp = client.post(
        "/v1/judge/pairwise",
        json={
            "bundle": {
                "question": "q",
                "answers": ["A answer", "B answer"],
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["winner"] == "A"
    # GLM-5.1-TEE rate card: $0.50/Mtok input, $2/Mtok output.
    # 1000 * 0.5 / 1M + 500 * 2 / 1M = 0.0005 + 0.001 = 0.0015
    assert body["cost_usd"] == pytest.approx(0.0015)


# -- /v1/judge/multi ----------------------------------------------------


def test_multi_response_includes_cost_usd(monkeypatch):
    _set_judge_env(monkeypatch)
    handler = _scripted_response(
        {
            "grounded_correctness": {"score": 0.9, "rationale": "ok"},
            "instruction_safety": {"score": 1.0, "rationale": "ok"},
        },
        prompt_tokens=2000, completion_tokens=200,
    )
    _patch_async_client(monkeypatch, "eiretes.eval.multi_judge", handler)

    client = TestClient(app)
    resp = client.post(
        "/v1/judge/multi",
        json={
            "bundle": {"question": "q", "answers": ["candidate"]},
            "applicable_metrics": ["grounded_correctness", "instruction_safety"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 2000 * 0.5 / 1M + 200 * 2 / 1M = 0.001 + 0.0004 = 0.0014
    assert body["cost_usd"] == pytest.approx(0.0014)


# -- /v1/judge/eval -----------------------------------------------------


def test_eval_response_includes_cost_usd(monkeypatch):
    _set_judge_env(monkeypatch)
    handler = _scripted_response(
        {"outcome": "correct", "failure_mode": None, "guidance": ""},
        prompt_tokens=1500, completion_tokens=100,
    )
    _patch_async_client(monkeypatch, "eiretes.eval.judge", handler)

    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval",
        json={
            "bundle": {"question": "q", "answers": ["candidate"]},
            "expected_answer": "expected",
            "oracle_source": "three_oracle",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 1500 * 0.5 / 1M + 100 * 2 / 1M = 0.00075 + 0.0002 = 0.00095
    assert body["cost_usd"] == pytest.approx(0.00095)


# -- Missing usage block falls back to None ------------------------------


def test_pairwise_cost_usd_is_none_when_usage_missing(monkeypatch):
    """Old/mocked upstreams may omit the ``usage`` block entirely. Cost
    must surface as ``None``, not silently zero — the validator
    distinguishes "unknown" from "$0 metered call" via the null."""
    _set_judge_env(monkeypatch)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(
                {"winner": "B", "confidence": 0.6, "reason": ""}
            )}}],
            # no ``usage`` field at all
        })

    _patch_async_client(monkeypatch, "eiretes.eval.pairwise", _handler)

    client = TestClient(app)
    resp = client.post(
        "/v1/judge/pairwise",
        json={"bundle": {"question": "q", "answers": ["x", "y"]}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["cost_usd"] is None
