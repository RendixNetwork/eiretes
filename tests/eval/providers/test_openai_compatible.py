"""HTTP-level tests for the eiretes-side OpenAI-compatible client.

Eiretes uses this client to call ``zai-org/GLM-5.1-TEE`` via Chutes.
Verifies the structured-output request shape, retry on transient
status codes, fail-fast on non-retryable 4xx, and timeout handling.
"""

from __future__ import annotations

import json

import httpx
import pytest

from eiretes.eval.config import JudgeConfig
from eiretes.eval.providers.openai_compatible import OpenAICompatibleClient
from eiretes.eval.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)


pytestmark = pytest.mark.asyncio


def _cfg(**overrides) -> JudgeConfig:
    base = dict(
        base_url="http://chutes.test",
        api_key="tok",
        model="zai-org/GLM-5.1-TEE",
        timeout_seconds=5.0,
        max_tokens=512,
    )
    base.update(overrides)
    return JudgeConfig(**base)


def _ok_response(content: str | list) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            # Token-count usage shape — Chutes' real ``/chat/completions``
            # response. Cost extraction multiplies by the rate card
            # (``zai-org/GLM-5.1-TEE`` = $0.5 / $2 per Mtok).
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        },
    )


async def test_complete_structured_returns_text_and_latency():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["url"] = str(request.url)
        return _ok_response(json.dumps({"outcome": "correct", "guidance": ""}))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(_cfg(), transport=transport)
    resp = await client.complete_structured(
        system="you are an eval judge",
        user=json.dumps({"prompt": "what is 2+2?"}),
        response_schema={
            "type": "object",
            "properties": {
                "outcome": {"type": "string"},
                "guidance": {"type": "string"},
            },
            "required": ["outcome"],
        },
        schema_name="eval_outcome",
    )
    await client.aclose()

    assert isinstance(resp, ProviderResponse)
    assert json.loads(resp.text) == {"outcome": "correct", "guidance": ""}
    assert resp.latency_ms >= 0
    # 1000 * 0.5 / 1M + 500 * 2 / 1M = 0.0005 + 0.001 = 0.0015
    assert resp.usage_usd == pytest.approx(0.0015)
    assert resp.finish_reason == "stop"
    assert captured["url"] == "http://chutes.test/chat/completions"
    body = captured["body"]
    assert body["model"] == "zai-org/GLM-5.1-TEE"
    assert body["response_format"]["json_schema"]["name"] == "eval_outcome"
    assert body["response_format"]["json_schema"]["strict"] is True


async def test_retry_on_transient_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "transient"})
        return _ok_response(json.dumps({"ok": True}))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _cfg(), transport=transport,
        max_retries=2, backoff_base_seconds=0.001,
    )
    resp = await client.complete_structured(
        system="s", user="u",
        response_schema={"type": "object"},
    )
    await client.aclose()
    assert calls["n"] == 2
    assert json.loads(resp.text) == {"ok": True}


async def test_non_retryable_4xx_raises_immediately():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _cfg(), transport=transport,
        max_retries=3, backoff_base_seconds=0.001,
    )
    with pytest.raises(ProviderError):
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()
    assert calls["n"] == 1


async def test_timeout_raises_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated")

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _cfg(), transport=transport,
        max_retries=1, backoff_base_seconds=0.001,
    )
    with pytest.raises(ProviderTimeout):
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()


async def test_unconfigured_raises_at_init():
    with pytest.raises(ProviderError):
        OpenAICompatibleClient(_cfg(api_key=""))
