"""Exact USD cost extraction for eiretes judge calls.

All three internal judge roles (``pairwise`` / ``multi`` / ``eval``)
share one Chutes-hosted GLM-5.1-TEE deployment via the OpenAI-
compatible ``/chat/completions`` endpoint. No web search, no special
tools — token cost is the entire bill.

Rates resolve via :mod:`eiretes.eval.providers.pricing` so an operator
can override Chutes pricing without redeploying eiretes (set
``EIRETES_LLM_PRICING_JSON``). Returns ``None`` when the upstream
``usage`` block is missing or empty so the caller treats the result as
"unknown" rather than zero.
"""

from __future__ import annotations

import logging
from typing import Any

from eiretes.eval.providers.pricing import cost_for, price_for

_logger = logging.getLogger(__name__)


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def extract_chutes_chat_cost(
    payload: dict[str, Any], model: str,
) -> float | None:
    """Compute exact USD cost for a Chutes ``/chat/completions`` call.

    Reads ``usage.prompt_tokens`` + ``usage.completion_tokens`` and
    multiplies by the configured rate card. Returns ``None`` when
    ``usage`` is missing entirely; ``0.0`` is reserved for the
    "fully-cached / zero-token" case which Chutes never actually
    emits.
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens"))
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    if price_for(model) is None:
        return None
    return round(cost_for(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    ), 8)


__all__ = ["extract_chutes_chat_cost"]
