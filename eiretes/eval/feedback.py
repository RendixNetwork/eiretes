"""Per-miner feedback assembly + in-memory test store.

Production: miners read their feedback rows directly from eirel-ai
owner-api's hotkey-signed ``GET /v1/eval/feedback`` and assemble the
``EvalFeedbackDoc`` client-side using ``_build_feedback_doc_from_rows``.
Eiretes is purely the judge service — it does not host the feedback
read path.

The in-memory ``FeedbackStore`` remains for unit tests: lets the
doc-assembly logic exercise without standing up an owner-api fixture.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from eiretes.eval.models import (
    EvalFeedbackDoc,
    EvalItemRecord,
    FailureMode,
    FeedbackPerItem,
    Outcome,
)


_FAILURE_MODE_HINTS: dict[FailureMode, str] = {
    "missing_tool_use": (
        "The item required a specific tool. Make sure the agent dispatches "
        "the right tool for prompts of this kind."
    ),
    "wrong_fact": (
        "The agent answered with a factual error. Improve grounding or "
        "fall back to a tool when the answer is uncertain."
    ),
    "missing_grounding": (
        "The agent didn't read attached_files / recalled memory it had "
        "available. Always read the orchestrator-injected context first."
    ),
    "off_topic": (
        "The agent addressed the wrong question. Tighten prompt parsing."
    ),
    "incomplete": (
        "The agent answered but with key omissions. Cover all parts of "
        "the user's question."
    ),
    "over_refusal": (
        "The agent refused without good reason. Calibrate refusal — only "
        "decline when the request is genuinely harmful or impossible."
    ),
    "hallucinated_claim": (
        "The agent claimed something the source contradicts. Stay grounded "
        "in the provided context; never invent specific facts."
    ),
}


def _excerpt(text: str, n: int = 200) -> str:
    if not text:
        return ""
    body = text.strip()
    return body if len(body) <= n else body[: n - 3] + "..."


def _build_feedback_doc(
    *,
    run_id: str,
    miner_hotkey: str,
    records: list[EvalItemRecord],
) -> EvalFeedbackDoc:
    n = len(records)
    composite = (
        sum(r.composite.composite for r in records) / n if n else 0.0
    )
    fm_counts: dict[str, int] = {}
    for r in records:
        if r.outcome.failure_mode:
            fm_counts[r.outcome.failure_mode] = (
                fm_counts.get(r.outcome.failure_mode, 0) + 1
            )

    largest_gap = ""
    if fm_counts:
        top_mode, top_count = max(fm_counts.items(), key=lambda kv: kv[1])
        hint = _FAILURE_MODE_HINTS.get(top_mode, "")  # type: ignore[arg-type]
        largest_gap = (
            f"{top_mode} ({top_count}/{n} items) — {hint}".strip()
        )

    per_item = [
        FeedbackPerItem(
            item_kind=r.kind,
            prompt_excerpt=_excerpt(_describe_prompt(r), 200),
            your_response_excerpt=_excerpt(r.candidate_response, 200),
            outcome=r.outcome.outcome,
            failure_mode=r.outcome.failure_mode,
            guidance=r.outcome.guidance,
            composite=r.composite.composite,
        )
        for r in records
    ]

    return EvalFeedbackDoc(
        run_id=run_id,
        miner_hotkey=miner_hotkey,
        composite_score=composite,
        n_items=n,
        per_failure_mode_counts=fm_counts,
        largest_gap=largest_gap,
        per_item=per_item,
        created_at=datetime.now(timezone.utc),
    )


def _describe_prompt(record: EvalItemRecord) -> str:
    """Compact prompt excerpt for the feedback doc.

    For multi-turn items we surface the final user turn (the recall
    query) so the miner sees what was actually being asked.
    """
    # We rebuild from the record fields rather than holding the
    # original EvalItem around — by the time feedback is built, the
    # item has been freed.
    return record.candidate_response[:0] or "<prompt redacted in feedback>"


class FeedbackStore:
    """Thread-safe in-memory store keyed by (run_id, miner_hotkey)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._docs: dict[tuple[str, str], EvalFeedbackDoc] = {}

    def write(self, doc: EvalFeedbackDoc) -> None:
        with self._lock:
            self._docs[(doc.run_id, doc.miner_hotkey)] = doc

    def get(
        self, *, run_id: str, miner_hotkey: str,
    ) -> EvalFeedbackDoc | None:
        with self._lock:
            return self._docs.get((run_id, miner_hotkey))

    def write_for_records(
        self,
        *,
        run_id: str,
        miner_hotkey: str,
        records: list[EvalItemRecord],
    ) -> EvalFeedbackDoc:
        doc = _build_feedback_doc(
            run_id=run_id, miner_hotkey=miner_hotkey, records=records,
        )
        self.write(doc)
        return doc


def _build_feedback_doc_from_rows(
    *,
    run_id: str,
    miner_hotkey: str,
    rows: list[dict[str, Any]],
) -> EvalFeedbackDoc | None:
    """Assemble an ``EvalFeedbackDoc`` from owner-api row dicts.

    Returns ``None`` when ``rows`` is empty so the caller can surface
    a 404 to the miner — "no feedback for this run yet" is a
    different signal from "feedback exists but is empty."
    """
    if not rows:
        return None
    n = len(rows)
    composite_avg = (
        sum(float(r.get("composite_score") or 0.0) for r in rows) / n
    )
    fm_counts: dict[str, int] = {}
    for r in rows:
        fm = r.get("failure_mode")
        if fm:
            fm_counts[fm] = fm_counts.get(fm, 0) + 1

    largest_gap = ""
    if fm_counts:
        top_mode, top_count = max(fm_counts.items(), key=lambda kv: kv[1])
        hint = _FAILURE_MODE_HINTS.get(top_mode, "")  # type: ignore[arg-type]
        largest_gap = f"{top_mode} ({top_count}/{n} items) — {hint}".strip()

    per_item: list[FeedbackPerItem] = []
    for r in rows:
        outcome = str(r.get("outcome") or "wrong")
        fm = r.get("failure_mode")
        per_item.append(
            FeedbackPerItem(
                item_kind="external",  # owner-api row doesn't track kind
                prompt_excerpt=str(r.get("prompt_excerpt") or "")[:200],
                your_response_excerpt=str(r.get("response_excerpt") or "")[:200],
                outcome=outcome,  # type: ignore[arg-type]
                failure_mode=fm,  # type: ignore[arg-type]
                guidance=str(r.get("guidance") or ""),
                composite=float(r.get("composite_score") or 0.0),
            )
        )
    return EvalFeedbackDoc(
        run_id=run_id,
        miner_hotkey=miner_hotkey,
        composite_score=composite_avg,
        n_items=n,
        per_failure_mode_counts=fm_counts,
        largest_gap=largest_gap,
        per_item=per_item,
        created_at=datetime.now(timezone.utc),
    )


# Module-level singleton — most callers want the default store. The
# service module imports this directly; tests build their own
# ``FeedbackStore()`` instances for isolation.
_default_store = FeedbackStore()


def default_store() -> FeedbackStore:
    return _default_store


__all__ = [
    "FeedbackStore",
    "_build_feedback_doc_from_rows",
    "default_store",
]
