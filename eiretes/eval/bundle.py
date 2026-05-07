"""Structured judge input — replaces raw ``prompt: str``.

Validator builds a ``JudgeInputBundle`` once per (task, miner) pair
in ``_judge_miner`` from the task fixture + miner response + cached
``expected_claims`` (from the validator's per-task oracle/reconciler
enrichment). Sends it over HTTP to eiretes; eiretes' per-role judge
modules call ``bundle.dispatch_for(role, budget_tokens)`` to render
the role-specific user-prompt JSON.

Greedy inclusion priority when budget is tight:
    question > answers > attached_summary > conversation_summary
       > conversation_recent > attached_full

Per-role rules layered on top of priority:
  * pairwise / multi never receive ``attached_full`` — pairwise judges
    preference (no facts), multi-judge consumes pre-extracted
    ``expected_claims`` instead.
  * eval may receive ``attached_full`` if budget allows AND the
    caller hasn't already extracted claims (rare — usually
    ``expected_claims`` are present).

Forward compat: new pool kinds add new fields to the bundle (e.g.
``code_repo_summary``, ``multimedia_descriptors``); the dispatcher's
priority list is the only thing that changes.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field


JudgeRole = Literal["pairwise", "multi", "eval"]


# Rough token-budget estimator used by ``dispatch_for``. ~4 chars per
# token is a coarse but OK-for-LLM-budgeting approximation. Production
# judges should pass ``budget_tokens`` derived from the model's actual
# context window minus reservations for system prompt + response.
_CHARS_PER_TOKEN = 4


_DEFAULT_BUDGET_TOKENS = 8000


class JudgeInputBundle(BaseModel):
    """Structured judge input — task-shape fields only.

    Per-call extras (``expected_answer``, ``expected_claims``,
    ``must_not_claim``, ``required_tool``, ``oracle_source``) are NOT
    part of the bundle — they're caller-supplied at the judge call
    site. The bundle covers fields that vary per (task, miner) pair.

    ``answers`` is a 1-element list for ``eval`` / ``multi`` (single
    candidate) or 2-element list for ``pairwise`` (A/B comparison).
    Caller is responsible for matching role to list length.
    """

    question: str
    attached_summary: str | None = None
    attached_full: str | None = None
    conversation_summary: str | None = None
    conversation_recent: list[dict[str, str]] = Field(default_factory=list)
    constraints: str | None = None
    answers: list[str] = Field(default_factory=list)

    def dispatch_for(
        self,
        *,
        role: JudgeRole,
        budget_tokens: int = _DEFAULT_BUDGET_TOKENS,
    ) -> dict[str, Any]:
        """Render the role-specific user-prompt fields, fitting budget.

        Returns a JSON-serializable dict the judge wraps + posts. The
        caller adds per-role extras (``expected_answer``,
        ``must_not_claim``, ``oracle_source``) on top — those don't
        live on the bundle.
        """
        out: dict[str, Any] = {"question": self.question}

        # Answers — schema differs per role. Always included.
        if role == "pairwise":
            if len(self.answers) != 2:
                raise ValueError(
                    f"pairwise role requires answers tuple of length 2; "
                    f"got {len(self.answers)}"
                )
            out["answer_a"] = self.answers[0]
            out["answer_b"] = self.answers[1]
        else:
            if len(self.answers) != 1:
                raise ValueError(
                    f"{role!r} role requires answers tuple of length 1; "
                    f"got {len(self.answers)}"
                )
            out["candidate_response"] = self.answers[0]

        # Constraints (must_not_claim floor / format directives) —
        # always included when present; small.
        if self.constraints:
            out["constraints"] = self.constraints

        # Greedy inclusion of optional fields by priority. Track
        # remaining budget AFTER the always-included fields so a
        # truncation only ever drops the lowest-priority items.
        remaining = budget_tokens - _approx_tokens(out)

        # Order matters: highest priority first. attached_full is
        # always last and is gated by role.
        prioritized: list[tuple[str, Any]] = [
            ("attached_summary", self.attached_summary),
            ("conversation_summary", self.conversation_summary),
            ("conversation_recent",
             list(self.conversation_recent) if self.conversation_recent else None),
        ]
        if role == "eval":
            prioritized.append(("attached_full", self.attached_full))
        # For pairwise + multi, attached_full is dropped by role policy
        # regardless of budget (memory: pairwise/multi don't need raw
        # docs; the judge sees a 200-token attached_summary instead).

        for key, value in prioritized:
            if not value:
                continue
            cost = _approx_tokens({key: value})
            if cost > remaining:
                continue
            out[key] = value
            remaining -= cost

        return out


# -- helpers --------------------------------------------------------------


def _approx_tokens(payload: Any) -> int:
    """Coarse token estimate via JSON length / 4. Off by a small
    multiplicative factor on text-heavy payloads but consistent enough
    for budget-fitting decisions."""
    return len(json.dumps(payload, ensure_ascii=False)) // _CHARS_PER_TOKEN


__all__ = [
    "JudgeInputBundle",
    "JudgeRole",
]
