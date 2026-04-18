"""Rubric catalog for the eiretes judge service.

`RUBRIC_CATALOG` is the single source of truth for how each execution family
is scored. After the clean-slate refactor there is only one launch family —
`general_chat` — judged on the four quality dimensions defined in
`rubrics/general_chat.py`. Future families (`deep_research`, `coding`) plug in
by adding entries here and a matching branch in
`LLMJudgeClient._dimension_scores`.
"""

from __future__ import annotations

from typing import Any

from .rubrics.general_chat import GENERAL_CHAT_QUALITY_RUBRIC

RUBRIC_CATALOG: dict[str, dict[str, Any]] = {
    "general_chat": GENERAL_CHAT_QUALITY_RUBRIC,
}

_VALID_MODES: frozenset[str] = frozenset({"instant", "thinking"})


def resolve_rubric_spec(
    family_id: str,
    *,
    mode: str = "instant",
    **_: Any,
) -> dict[str, Any]:
    """Resolve a rubric spec for a given family_id and mode.

    Returns a shallow copy of the catalog entry so callers can mutate fields
    like ``active_mode`` and ``active_system_prompt`` without corrupting the
    shared catalog.
    """
    family_id = str(family_id).strip()
    if family_id not in RUBRIC_CATALOG:
        raise ValueError(
            f"unknown family_id {family_id!r}; valid families: {sorted(RUBRIC_CATALOG)}"
        )
    resolved_mode = str(mode or "instant").strip().lower()
    if resolved_mode not in _VALID_MODES:
        raise ValueError(
            f"unknown mode {resolved_mode!r}; expected one of {sorted(_VALID_MODES)}"
        )
    spec = dict(RUBRIC_CATALOG[family_id])
    system_prompt_by_mode: dict[str, str] = dict(spec.get("system_prompt_by_mode") or {})
    spec["active_mode"] = resolved_mode
    spec["active_system_prompt"] = system_prompt_by_mode.get(resolved_mode, "")
    return spec
