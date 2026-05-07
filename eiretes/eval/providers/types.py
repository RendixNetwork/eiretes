"""Shared response/error types for eiretes-side LLM provider client."""

from __future__ import annotations

from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Generic provider failure that callers (judges) should surface as
    a malformed-judge-response signal."""


class ProviderTimeout(ProviderError):
    """Subclass for timeouts and exhausted-retry network errors."""


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized completion response.

    ``text`` is the raw structured-output payload; the judge module
    decodes JSON. ``finish_reason`` is informational. ``usage_usd`` is
    best-effort — Chutes responses don't always include cost.
    """

    text: str
    latency_ms: int
    usage_usd: float | None = None
    finish_reason: str | None = None
