"""Chutes-side pricing table for eiretes judges.

Eiretes runs the three internal judge roles (``pairwise`` / ``multi``
/ ``eval``) against Chutes-hosted ``zai-org/GLM-5.1-TEE`` (TEE-attestable).
This module mirrors the ``shared.common.tool_pricing`` table from
``eirel-ai`` for the Chutes models eiretes actually calls — eiretes
is a separate package and can't import from ``eirel-ai``, so the
relevant entries are duplicated here.

Rates come from ``https://llm.chutes.ai/v1/models``
(``price.input.usd`` / ``price.output.usd``, both in $/1M tokens).
The ``chutes:*`` fallback is intentionally generous so unfamiliar
models aren't silently undercharged. Override at runtime via
``EIRETES_LLM_PRICING_JSON`` (same shape as ``EIREL_LLM_PRICING_JSON``)
when Chutes publishes a rate change before this file is updated.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class LLMPrice:
    input_per_mtok_usd: float
    output_per_mtok_usd: float


_DEFAULT_PRICING: dict[str, LLMPrice] = {
    "chutes:zai-org/GLM-5.1-TEE": LLMPrice(0.50, 2.0),
    "chutes:moonshotai/Kimi-K2.5-TEE": LLMPrice(0.3827, 1.72),
    "chutes:Qwen/Qwen3-32B-TEE": LLMPrice(0.08, 0.24),
    "chutes:MiniMaxAI/MiniMax-M2.5-TEE": LLMPrice(0.118, 0.99),
    "chutes:*": LLMPrice(0.50, 2.0),
}


def _load_pricing() -> dict[str, LLMPrice]:
    pricing = dict(_DEFAULT_PRICING)
    raw = os.getenv("EIRETES_LLM_PRICING_JSON")
    if raw:
        try:
            overrides = json.loads(raw)
            for key, entry in overrides.items():
                pricing[key] = LLMPrice(
                    input_per_mtok_usd=entry["input_per_mtok_usd"],
                    output_per_mtok_usd=entry["output_per_mtok_usd"],
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            _logger.warning(
                "invalid EIRETES_LLM_PRICING_JSON, using defaults",
            )
    return pricing


PRICING: dict[str, LLMPrice] = _load_pricing()


def price_for(model: str) -> LLMPrice | None:
    """Resolve the Chutes rate card for ``model``. Falls back to the
    ``chutes:*`` glob when the exact key is unknown."""
    key = f"chutes:{model}"
    return PRICING.get(key) or PRICING.get("chutes:*")


def cost_for(*, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD token cost. Returns 0.0 when the model is unknown — caller
    can decide to log instead of charging the generous fallback."""
    price = price_for(model)
    if price is None:
        return 0.0
    return (
        prompt_tokens * price.input_per_mtok_usd
        + completion_tokens * price.output_per_mtok_usd
    ) / 1_000_000


__all__ = ["LLMPrice", "PRICING", "cost_for", "price_for"]
