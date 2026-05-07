"""Eiretes-side LLM provider client.

A single OpenAI-compatible client serves all three internal judge
roles (``pairwise``, ``multi``, ``eval``) — each calls the same
Chutes-hosted ``zai-org/GLM-5.1-TEE`` model with role-specific
prompts and schemas.

This module is intentionally separate from eirel-ai's validator-side
provider clients (see ``eirel-ai/validation/validator/providers/``).
The two repos do not share Python source; duplicating the minimal
client is cheaper than coordinating shared-library version bumps
across repo boundaries.
"""

from __future__ import annotations

from eiretes.eval.providers.openai_compatible import OpenAICompatibleClient
from eiretes.eval.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)

__all__ = [
    "OpenAICompatibleClient",
    "ProviderError",
    "ProviderResponse",
    "ProviderTimeout",
]
