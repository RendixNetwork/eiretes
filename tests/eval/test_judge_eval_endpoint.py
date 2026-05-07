"""HTTP-level tests for /v1/judge/eval and /v1/judge/eval/composite.

The validator engine in eirel-ai posts to these endpoints; eiretes
stays a pure judge service — no pool fetch, no dispatch, no runner.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from eiretes.service import app


def _stub_judge_handler(scripted: dict):
    """Build an httpx.MockTransport handler that returns ``scripted`` JSON."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(scripted)}}
                ]
            },
        )

    return _handler


# -- /v1/judge/eval --------------------------------------------------------


def test_judge_eval_returns_outcome_for_correct_answer(monkeypatch):
    monkeypatch.setenv("EIREL_EVAL_JUDGE_BASE_URL", "http://judge.test")
    monkeypatch.setenv("EIREL_EVAL_JUDGE_API_KEY", "tok")

    handler = _stub_judge_handler({
        "outcome": "correct",
        "failure_mode": None,
        "guidance": "ok",
    })
    transport = httpx.MockTransport(handler)
    _real_client = httpx.AsyncClient

    class _StubAsyncClient(_real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "eiretes.eval.judge.httpx.AsyncClient", _StubAsyncClient,
    )

    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval",
        json={
            "bundle": {
                "question": "What is 2 + 2?",
                "answers": ["The answer is 4."],
            },
            "expected_answer": "4",
            "must_not_claim": [],
            "oracle_source": "three_oracle",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["outcome"] == "correct"
    assert body["failure_mode"] is None


def test_judge_eval_handles_multi_turn_payload(monkeypatch):
    monkeypatch.setenv("EIREL_EVAL_JUDGE_BASE_URL", "http://judge.test")
    monkeypatch.setenv("EIREL_EVAL_JUDGE_API_KEY", "tok")

    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({
                        "outcome": "correct", "guidance": "",
                    })}}
                ]
            },
        )

    transport = httpx.MockTransport(_handler)
    _real_client = httpx.AsyncClient

    class _StubAsyncClient(_real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "eiretes.eval.judge.httpx.AsyncClient", _StubAsyncClient,
    )

    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval",
        json={
            "bundle": {
                "question": "What language do I work in?",
                "conversation_recent": [
                    {"role": "user", "content": "I work in Python."},
                    {"role": "assistant", "content": "Got it."},
                    {"role": "user", "content": "What language do I work in?"},
                ],
                "answers": ["You work in Python."],
            },
            "expected_answer": "Python",
            "oracle_source": "deterministic",
        },
    )
    assert resp.status_code == 200
    user_msg = next(
        m for m in captured["body"]["messages"] if m["role"] == "user"
    )
    payload = json.loads(user_msg["content"])
    assert payload["conversation_recent"] is not None
    assert len(payload["conversation_recent"]) == 3


def test_judge_eval_disputed_downgrades_for_deterministic(monkeypatch):
    """Disputed is only valid for three_oracle items; deterministic items
    where the candidate disagrees with the planted answer are wrong."""
    monkeypatch.setenv("EIREL_EVAL_JUDGE_BASE_URL", "http://judge.test")
    monkeypatch.setenv("EIREL_EVAL_JUDGE_API_KEY", "tok")

    handler = _stub_judge_handler({"outcome": "disputed", "guidance": ""})
    transport = httpx.MockTransport(handler)
    _real_client = httpx.AsyncClient

    class _StubAsyncClient(_real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "eiretes.eval.judge.httpx.AsyncClient", _StubAsyncClient,
    )

    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval",
        json={
            "bundle": {
                "question": "I work in Python.",
                "conversation_recent": [
                    {"role": "user", "content": "I work in Python."},
                ],
                "answers": ["You work in Rust."],
            },
            "expected_answer": "Python",
            "oracle_source": "deterministic",
        },
    )
    assert resp.status_code == 200
    # Downgraded from disputed → wrong (deterministic source IS truth).
    assert resp.json()["outcome"] == "wrong"


# -- /v1/judge/eval/composite ---------------------------------------------


def test_composite_endpoint_correct_no_tool_yields_one():
    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval/composite",
        json={
            "outcome": "correct",
            "candidate_response": "The answer is 42.",
            "must_not_claim": [],
            "required_tool": None,
            "ledger_tools": [],
            "latency_ms": 100,
            "cost_usd": 0.001,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["composite"] == pytest.approx(1.0, rel=1e-6)
    assert body["outcome_score"] == 1.0
    assert body["tool_attestation_factor"] == 1.0
    assert body["knockout_reason"] is None


def test_composite_endpoint_zeroes_when_required_tool_missing():
    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval/composite",
        json={
            "outcome": "correct",
            "candidate_response": "ok",
            "must_not_claim": [],
            "required_tool": "web_search",
            "ledger_tools": [],
            "latency_ms": 100,
            "cost_usd": 0.001,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["composite"] == 0.0
    assert body["tool_attestation_factor"] == 0.0


def test_composite_endpoint_zeroes_on_must_not_claim():
    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval/composite",
        json={
            "outcome": "correct",
            "candidate_response": "Cats have six legs.",
            "must_not_claim": ["six"],
            "required_tool": None,
            "ledger_tools": [],
            "latency_ms": 100,
            "cost_usd": 0.001,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["composite"] == 0.0
    assert body["hallucination_knockout"] == 0.0
    assert "must_not_claim" in (body["knockout_reason"] or "")


def test_composite_endpoint_zero_cost_floors_composite():
    client = TestClient(app)
    resp = client.post(
        "/v1/judge/eval/composite",
        json={
            "outcome": "correct",
            "candidate_response": "ok",
            "must_not_claim": [],
            "required_tool": None,
            "ledger_tools": [],
            "latency_ms": 100,
            "cost_usd": 0.0,
            "cost_floor_usd": 0.00005,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["composite"] == 0.0
    assert body["cost_attestation_knockout"] == 0.0
