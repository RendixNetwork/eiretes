"""Rubric catalog for the eiretes judge service.

After the pairwise redesign, each family is judged by comparing a candidate
response against a baseline response (typically OpenAI Responses API with
built-in web_search). The catalog entry carries the rubric name, the four
dimensions scored on each side, and the single system prompt the judge uses.
"""

from __future__ import annotations

from typing import Any

from .rubrics.pairwise_general_chat import PAIRWISE_GENERAL_CHAT_RUBRIC

RUBRIC_CATALOG: dict[str, dict[str, Any]] = {
    "general_chat": PAIRWISE_GENERAL_CHAT_RUBRIC,
}


def resolve_rubric_spec(family_id: str) -> dict[str, Any]:
    """Resolve a rubric spec for a given family_id.

    Returns a shallow copy so callers can mutate transient fields without
    corrupting the shared catalog.
    """
    family_id = str(family_id).strip()
    if family_id not in RUBRIC_CATALOG:
        raise ValueError(
            f"unknown family_id {family_id!r}; valid families: {sorted(RUBRIC_CATALOG)}"
        )
    return dict(RUBRIC_CATALOG[family_id])
