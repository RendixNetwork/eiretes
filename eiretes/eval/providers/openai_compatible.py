"""Minimal OpenAI-compatible chat-completions client for Chutes judges.

Used by the three internal judge roles (``pairwise``, ``multi``,
``eval``) to call ``zai-org/GLM-5.1-TEE`` via Chutes' OpenAI-compatible
endpoint. Single-file implementation — eiretes' total external surface
is one provider, so a generic abstraction would be over-engineering.

Hardening:
  * Async ``httpx`` with explicit timeout per call.
  * Bounded retry on 429/502/503/504; other 4xx surface immediately as
    :class:`ProviderError`.
  * ``response_format={"type": "json_schema", ...}`` enforces structured
    output; caller decodes JSON.
  * Cost extraction is best-effort.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from eiretes.eval.config import JudgeConfig
from eiretes.eval.providers.cost_calc import extract_chutes_chat_cost
from eiretes.eval.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)

_logger = logging.getLogger(__name__)


_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE_SECONDS = 0.5


class OpenAICompatibleClient:
    """Thin wrapper around ``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        cfg: JudgeConfig,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not cfg.configured:
            raise ProviderError(
                "OpenAICompatibleClient requires base_url + api_key + model"
            )
        self._cfg = cfg
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = max(0.0, float(backoff_base_seconds))
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def base_url(self) -> str:
        return self._cfg.base_url

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema_name: str = "response",
    ) -> ProviderResponse:
        """Single chat-completions call with strict structured output."""
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": response_schema,
                    "strict": True,
                },
            },
            "temperature": float(temperature),
            "max_tokens": int(max_tokens or self._cfg.max_tokens),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        url = f"{self._cfg.base_url}/chat/completions"
        client = await self._get_client()
        return await self._post_with_retry(
            client=client, url=url, payload=payload, headers=headers,
        )

    async def _post_with_retry(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ProviderResponse:
        last_exc: Exception | None = None
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._cfg.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt > self._max_retries:
                    raise ProviderTimeout(
                        f"timeout after {attempt} attempt(s): {exc}"
                    ) from exc
                await self._sleep_backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt > self._max_retries:
                    raise ProviderError(
                        f"network error after {attempt} attempt(s): {exc}"
                    ) from exc
                await self._sleep_backoff(attempt)
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if response.status_code in _RETRY_STATUSES and attempt <= self._max_retries:
                await self._sleep_backoff(attempt)
                continue
            if response.status_code != 200:
                raise ProviderError(
                    f"HTTP {response.status_code}: "
                    f"{(response.text or '')[:512]}"
                )
            parsed = self._parse_response(response, latency_ms)
            cost = parsed.usage_usd
            cost_str = "?" if cost is None else f"${cost:.6f}"
            _logger.info(
                "judge_provider_call: model=%s latency_ms=%d cost_usd=%s",
                self._cfg.model, latency_ms, cost_str,
            )
            return parsed
        raise ProviderError(  # pragma: no cover
            f"unreachable after {attempt} attempts; last_exc={last_exc!r}"
        )

    async def _sleep_backoff(self, attempt: int) -> None:
        delay = self._backoff_base * (2 ** (attempt - 1))
        await asyncio.sleep(delay)

    def _parse_response(
        self, response: httpx.Response, latency_ms: int,
    ) -> ProviderResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"non-JSON response: {exc}") from exc
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(
                f"missing choices in response: {str(payload)[:512]}"
            )
        choice = choices[0] or {}
        message = choice.get("message") or {}
        raw_content = message.get("content")
        if isinstance(raw_content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            text = str(raw_content or "")
        finish_reason = choice.get("finish_reason")
        usage_usd = self._extract_usage_usd(payload)
        return ProviderResponse(
            text=text,
            latency_ms=latency_ms,
            usage_usd=usage_usd,
            finish_reason=finish_reason,
        )

    def _extract_usage_usd(self, payload: dict[str, Any]) -> float | None:
        """Compute exact USD cost for the call.

        Eiretes only ever calls Chutes-hosted models via this client,
        so the dispatch is unconditional. ``EIRETES_LLM_PRICING_JSON``
        overrides the static rate table when ops need to pin a
        per-model rate without a redeploy.
        """
        return extract_chutes_chat_cost(payload, self._cfg.model)
