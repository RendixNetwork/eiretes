"""Outcome alias + soft-fail tests for the eval judge.

GLM-5.1-TEE occasionally returns a failure_mode value (``"incomplete"``,
``"over_refusal"``) where the rubric expects an outcome enum value.
Pre-0.2.2 the judge raised ``ValueError`` and the validator's whole
miner-judgment cycle 500'd. The alias map preserves the model's
intent (incomplete answer = partial credit + failure_mode=incomplete);
the soft-fail fallback catches future drift without breaking runs.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from eiretes.service import app


def _stub(scripted: dict):
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(scripted)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })
    return _handler


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    class _StubAsyncClient(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("eiretes.eval.judge.httpx.AsyncClient", _StubAsyncClient)


def _set_env(monkeypatch):
    monkeypatch.setenv("EIREL_EVAL_JUDGE_BASE_URL", "https://llm.chutes.ai/v1")
    monkeypatch.setenv("EIREL_EVAL_JUDGE_API_KEY", "tok")
    monkeypatch.setenv("EIREL_EVAL_JUDGE_MODEL", "zai-org/GLM-5.1-TEE")


def _post(client: TestClient, body_outcome: dict) -> dict:
    return client.post("/v1/judge/eval", json={
        "bundle": {"question": "q", "answers": ["candidate"]},
        "expected_answer": "expected",
        "oracle_source": "three_oracle",
    }).json()


# -- Aliases preserve model intent ------------------------------------------


def test_incomplete_outcome_maps_to_partial(monkeypatch):
    """The exact production drift: outcome='incomplete' → outcome=partial,
    failure_mode=incomplete."""
    _set_env(monkeypatch)
    _patch_client(monkeypatch, _stub({
        "outcome": "incomplete",
        "failure_mode": None,
        "guidance": "missing detail",
    }))
    body = _post(TestClient(app), {})
    assert body["outcome"] == "partial"
    assert body["failure_mode"] == "incomplete"


def test_over_refusal_outcome_maps_to_refused(monkeypatch):
    _set_env(monkeypatch)
    _patch_client(monkeypatch, _stub({
        "outcome": "over_refusal",
        "failure_mode": None,
        "guidance": "",
    }))
    body = _post(TestClient(app), {})
    assert body["outcome"] == "refused"
    assert body["failure_mode"] == "over_refusal"


def test_hallucinated_claim_outcome_maps_to_hallucinated(monkeypatch):
    _set_env(monkeypatch)
    _patch_client(monkeypatch, _stub({
        "outcome": "hallucinated_claim",
        "failure_mode": None,
        "guidance": "",
    }))
    body = _post(TestClient(app), {})
    assert body["outcome"] == "hallucinated"
    assert body["failure_mode"] == "hallucinated_claim"


def test_alias_does_not_overwrite_explicit_failure_mode(monkeypatch):
    """If the model emitted both an aliased outcome AND a different
    failure_mode, keep the explicit failure_mode rather than
    overwriting with the alias's implied one."""
    _set_env(monkeypatch)
    _patch_client(monkeypatch, _stub({
        "outcome": "incomplete",  # → would imply failure_mode=incomplete
        "failure_mode": "missing_tool_use",  # but model said this
        "guidance": "",
    }))
    body = _post(TestClient(app), {})
    assert body["outcome"] == "partial"
    # Explicit failure_mode wins over the alias's default.
    assert body["failure_mode"] == "missing_tool_use"


# -- Soft-fail for truly unknown values -------------------------------------


def test_unknown_outcome_soft_fails_to_wrong(monkeypatch):
    """An outcome value we've never seen → ``wrong`` with no
    failure_mode. No 500, no ValueError — the run continues."""
    _set_env(monkeypatch)
    _patch_client(monkeypatch, _stub({
        "outcome": "totally-made-up-verdict",
        "failure_mode": None,
        "guidance": "",
    }))
    body = _post(TestClient(app), {})
    assert body["outcome"] == "wrong"
    assert body["failure_mode"] is None


def test_known_outcome_unchanged(monkeypatch):
    """Sanity: canonical outcomes pass through untouched."""
    _set_env(monkeypatch)
    _patch_client(monkeypatch, _stub({
        "outcome": "correct",
        "failure_mode": None,
        "guidance": "ok",
    }))
    body = _post(TestClient(app), {})
    assert body["outcome"] == "correct"
    assert body["failure_mode"] is None


def test_unknown_failure_mode_still_soft_fails(monkeypatch):
    """Existing behaviour preserved: unknown failure_mode → None."""
    _set_env(monkeypatch)
    _patch_client(monkeypatch, _stub({
        "outcome": "wrong",
        "failure_mode": "made-up-mode",
        "guidance": "",
    }))
    body = _post(TestClient(app), {})
    assert body["outcome"] == "wrong"
    assert body["failure_mode"] is None
