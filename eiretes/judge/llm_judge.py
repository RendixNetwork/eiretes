from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx
from pydantic import ValidationError

from eiretes.judge.catalog import RUBRIC_CATALOG, resolve_rubric_spec
from eiretes.models import JudgeResult, ProviderJudgeResponse
from eiretes.utils import float_env

_logger = logging.getLogger(__name__)

_PROVIDER_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    ValueError,
    ValidationError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
)

# -- Deterministic fallback signal vocab --------------------------------------
# Externalised so the keyword sets can be audited and tuned in one place rather
# than scattered across _dimension_scores. Weights remain inline with each
# dimension formula because they're not reusable.

_CITATION_MARKERS: frozenset[str] = frozenset({"http://", "https://"})
_EVIDENCE_MARKERS: frozenset[str] = frozenset(
    {"evidence", "source", "according to", "reports", "study", "found"}
)
_HEDGING_MARKERS: frozenset[str] = frozenset(
    {"it depends", "might", "possibly", "perhaps", "likely", "uncertain"}
)
_COMMITMENT_MARKERS: frozenset[str] = frozenset(
    {"recommend", "the answer is", "you should", "in summary", "conclusion"}
)
_PREAMBLE_MARKERS: frozenset[str] = frozenset(
    {
        "certainly",
        "i'd be happy",
        "i would be happy",
        "great question",
        "happy to help",
        "of course",
        "absolutely",
        "let me",
    }
)
_CONTRADICTION_MARKERS: frozenset[str] = frozenset(
    {"however actually", "but actually", "on the other hand i was wrong", "scratch that"}
)


def _contains_any(haystack: str, markers: frozenset[str]) -> bool:
    return any(token in haystack for token in markers)


def _count_any(haystack: str, markers: frozenset[str]) -> int:
    return sum(haystack.count(token) for token in markers)


class LLMJudgeClient:
    def __init__(
        self,
        *,
        model: str,
        rubric_version: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        deterministic_fallback: bool = True,
        ensemble_base_url: str | None = None,
        ensemble_api_key: str | None = None,
        ensemble_timeout_seconds: float | None = None,
        ensemble_disagreement_threshold: float | None = None,
    ):
        self.model = model
        self.rubric_version = rubric_version
        self.base_url = (base_url or os.getenv("EIREL_JUDGE_BASE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("EIREL_JUDGE_API_KEY", "")
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float_env("EIREL_JUDGE_TIMEOUT_SECONDS", 30.0, minimum=0.1)
        )
        self.transport = transport
        self.deterministic_fallback = deterministic_fallback
        self.ensemble_base_url = (
            (ensemble_base_url or os.getenv("EIREL_ENSEMBLE_JUDGE_BASE_URL", "")).rstrip("/")
        )
        self.ensemble_api_key = (
            ensemble_api_key
            if ensemble_api_key is not None
            else os.getenv("EIREL_ENSEMBLE_JUDGE_API_KEY", "")
        )
        self.ensemble_timeout_seconds = (
            float(ensemble_timeout_seconds)
            if ensemble_timeout_seconds is not None
            else float_env("EIREL_ENSEMBLE_JUDGE_TIMEOUT_SECONDS", 30.0, minimum=0.1)
        )
        self.ensemble_disagreement_threshold = (
            float(ensemble_disagreement_threshold)
            if ensemble_disagreement_threshold is not None
            else float_env(
                "EIREL_ENSEMBLE_JUDGE_DISAGREEMENT_THRESHOLD", 0.20, minimum=0.0, maximum=1.0
            )
        )
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                # Timeout is applied per-request to allow primary/ensemble to differ.
                self._client = httpx.AsyncClient(transport=self.transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def judge(
        self,
        *,
        family_id: str,
        prompt: str,
        response_excerpt: str,
        rubric_variant: str | None = None,
        mode: str = "instant",
    ) -> JudgeResult:
        """Score a miner response against the family rubric.

        Runs the primary provider; when an ensemble endpoint is configured it
        fans out primary and secondary calls concurrently and averages only
        dimensions both judges actually emitted. On provider failure the
        deterministic token-signal fallback runs and `metadata["provider_used"]`
        is set to `False`. Safe to call concurrently from multiple tasks — the
        underlying `httpx.AsyncClient` is shared and created lazily.
        """
        family_id = str(family_id).strip()
        if not (self.ensemble_base_url and self.ensemble_api_key):
            return await self._judge_primary(
                family_id=family_id,
                prompt=prompt,
                response_excerpt=response_excerpt,
                rubric_variant=rubric_variant,
                mode=mode,
            )
        # Primary and ensemble share the same provider payload — fan them out
        # concurrently so ensemble doesn't double the wall-clock latency.
        primary_task = asyncio.create_task(
            self._judge_primary(
                family_id=family_id,
                prompt=prompt,
                response_excerpt=response_excerpt,
                rubric_variant=rubric_variant,
                mode=mode,
            )
        )
        secondary_task = asyncio.create_task(
            self._judge_via_provider_at(
                base_url=self.ensemble_base_url,
                api_key=self.ensemble_api_key,
                timeout_seconds=self.ensemble_timeout_seconds,
                family_id=family_id,
                prompt=prompt,
                response_excerpt=response_excerpt,
                rubric_variant=rubric_variant,
                mode=mode,
            )
        )
        primary = await primary_task
        try:
            secondary = await secondary_task
        except _PROVIDER_EXCEPTIONS as exc:
            # Ensemble failure must never block scoring — log and return primary
            _logger.warning("ensemble judge failed, falling back to primary: %s", exc)
            return primary

        primary_keys = set(primary.dimension_scores)
        secondary_keys = set(secondary.dimension_scores)
        shared_keys = primary_keys & secondary_keys
        # Only average dimensions both judges actually computed. Keys present in
        # only one judge fall back to that judge's value so we don't discard signal.
        averaged_dimensions: dict[str, float] = {
            k: (primary.dimension_scores[k] + secondary.dimension_scores[k]) / 2.0
            for k in shared_keys
        }
        for k in primary_keys - secondary_keys:
            averaged_dimensions[k] = primary.dimension_scores[k]
        for k in secondary_keys - primary_keys:
            averaged_dimensions[k] = secondary.dimension_scores[k]

        averaged_score = (primary.score + secondary.score) / 2.0
        disagreement = abs(primary.score - secondary.score)
        per_dimension_delta = {
            k: round(abs(primary.dimension_scores[k] - secondary.dimension_scores[k]), 4)
            for k in shared_keys
        }
        flags = list(primary.constraint_flags)
        if disagreement >= self.ensemble_disagreement_threshold:
            flags.append(f"judge_disagreement:{disagreement:.2f}")
        return JudgeResult(
            model=f"{primary.model}+ensemble",
            rubric_name=primary.rubric_name,
            score=max(0.0, min(1.0, averaged_score)),
            rationale=primary.rationale,
            latency_seconds=primary.latency_seconds,
            dimension_scores=averaged_dimensions,
            constraint_flags=flags,
            usage={**dict(primary.usage or {}), **dict(secondary.usage or {})},
            metadata={
                **primary.metadata,
                "ensemble_agreement": round(1.0 - disagreement, 4),
                "primary_score": primary.score,
                "secondary_score": secondary.score,
                "ensemble_used": True,
                "dimension_coverage_mismatch": {
                    "primary_only": sorted(primary_keys - secondary_keys),
                    "secondary_only": sorted(secondary_keys - primary_keys),
                },
                "per_dimension_delta": per_dimension_delta,
            },
        )

    async def _judge_primary(
        self,
        *,
        family_id: str,
        prompt: str,
        response_excerpt: str,
        rubric_variant: str | None,
        mode: str = "instant",
    ) -> JudgeResult:
        if self.base_url and self.api_key:
            try:
                return await self._judge_via_provider_at(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout_seconds=self.timeout_seconds,
                    family_id=family_id,
                    prompt=prompt,
                    response_excerpt=response_excerpt,
                    rubric_variant=rubric_variant,
                    mode=mode,
                )
            except _PROVIDER_EXCEPTIONS as exc:
                if not self.deterministic_fallback:
                    raise
                _logger.warning(
                    "primary judge failed, using deterministic fallback: %s", exc
                )
                fallback = self._deterministic_judge(
                    family_id=family_id,
                    prompt=prompt,
                    response_excerpt=response_excerpt,
                    rubric_variant=rubric_variant,
                    mode=mode,
                )
                fallback.metadata = {
                    **fallback.metadata,
                    "provider_error": str(exc),
                    "provider_used": False,
                }
                return fallback
        return self._deterministic_judge(
            family_id=family_id,
            prompt=prompt,
            response_excerpt=response_excerpt,
            rubric_variant=rubric_variant,
            mode=mode,
        )

    @staticmethod
    def _format_dimension_rubrics(dimension_rubrics: dict[str, Any], dimensions: list[str]) -> str:
        lines: list[str] = [
            "SCORING RUBRIC — per-dimension anchors (scores are integers 1–5, mapped to 0.2/0.4/0.6/0.8/1.0):"
        ]
        for dim in dimensions:
            if dim not in dimension_rubrics:
                continue
            rubric = dimension_rubrics[dim]
            lines.append(f"\n[{dim}]")
            lines.append(f"  What it measures: {rubric.get('description', '')}")
            lines.append(f"  5 (exemplary):  {rubric.get('score_5', '')}")
            lines.append(f"  4 (good):       {rubric.get('score_4', '')}")
            lines.append(f"  3 (partial):    {rubric.get('score_3', '')}")
            lines.append(f"  2 (weak):       {rubric.get('score_2', '')}")
            lines.append(f"  1 (poor):       {rubric.get('score_1', '')}")
            penalties = rubric.get("penalties") or []
            if penalties:
                lines.append(
                    f"  Penalties (apply −0.2 per flag): {'; '.join(str(p) for p in penalties)}"
                )
        return "\n".join(lines)

    async def _judge_via_provider_at(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        family_id: str,
        prompt: str,
        response_excerpt: str,
        rubric_variant: str | None,
        mode: str = "instant",
    ) -> JudgeResult:
        """POST the judge payload to an OpenAI-compatible `/chat/completions`.

        The provider is expected to return a JSON chat completion whose
        `choices[0].message.content` is a JSON object matching
        `ProviderJudgeResponse` (overall_score in [0,1], dimension_scores as a
        numeric object, rationale string, constraint_flags list). Any schema
        violation raises `ValueError`, which bubbles up to `_judge_primary` where
        the deterministic fallback takes over.
        """
        spec = resolve_rubric_spec(family_id, mode=mode)
        rubric_name = str(spec["rubric_name"])
        effective_rubric_name = f"{rubric_name}:{self.rubric_version}"
        if rubric_variant:
            effective_rubric_name = f"{effective_rubric_name}:{rubric_variant}"
        dimensions = [str(item) for item in spec.get("dimensions", [])]
        response_format = {
            "type": "json_object",
        }
        system_prompt = str(spec.get("active_system_prompt") or "").strip()
        dimension_rubrics: dict[str, Any] = dict(spec.get("dimension_rubrics") or {})  # type: ignore[arg-type]
        if dimension_rubrics:
            formatted_rubrics = self._format_dimension_rubrics(dimension_rubrics, dimensions)
            system_prompt = (
                f"{system_prompt}\n\n"
                f"{formatted_rubrics}\n\n"
                "Score each dimension independently using the anchors above. "
                "Convert your integer anchor (1–5) to a decimal score: "
                "1→0.2, 2→0.4, 3→0.6, 4→0.8, 5→1.0. "
                "Apply penalty deductions (−0.2 each, floor 0.0) for any listed penalty patterns observed. "
            ).strip()
        system_prompt = (
            f"{system_prompt} "
            "Return strict JSON with keys overall_score, dimension_scores, constraint_flags, rationale. "
            "overall_score must be a number from 0 to 1. "
            "dimension_scores must contain numeric values from 0 to 1 using exactly these keys: "
            f"{', '.join(dimensions) if dimensions else 'quality'}. "
            "constraint_flags must be a list of short strings describing any constraint violations or penalty triggers observed. "
            "rationale must be concise and reference specific observations for each dimension."
        ).strip()
        user_prompt = json.dumps(
            {
                "rubric_name": effective_rubric_name,
                "family_id": family_id,
                "rubric_variant": rubric_variant,
                "mode": spec.get("active_mode"),
                "dimensions": dimensions,
                "prompt": prompt,
                "response_excerpt": response_excerpt,
            },
            sort_keys=True,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": response_format,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        started = time.perf_counter()
        client = await self._get_client()
        response = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        parsed = response.json()
        latency = max(0.0, time.perf_counter() - started)
        choice = ((parsed.get("choices") or [{}])[0] or {})
        message = choice.get("message") or {}
        raw_content = message.get("content") or "{}"
        if isinstance(raw_content, list):
            raw_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        try:
            judged_raw = json.loads(str(raw_content))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM judge returned malformed JSON (model={self.model}): {exc}"
            ) from exc
        try:
            judged = ProviderJudgeResponse.model_validate(judged_raw)
        except ValidationError as exc:
            raise ValueError(
                f"LLM judge returned invalid schema (model={self.model}): {exc}"
            ) from exc
        return JudgeResult(
            model=self.model,
            rubric_name=effective_rubric_name,
            score=judged.overall_score,
            rationale=judged.rationale or "Provider-backed general_chat judge.",
            latency_seconds=latency,
            dimension_scores=judged.dimension_scores,
            constraint_flags=judged.constraint_flags,
            usage=dict(parsed.get("usage") or {}),
            metadata={
                "family_id": family_id,
                "rubric_version": self.rubric_version,
                "rubric_variant": rubric_variant,
                "mode": spec.get("active_mode"),
                "provider_used": True,
                "judge_weight": float(spec["judge_weight"]),
            },
        )

    def _deterministic_judge(
        self,
        *,
        family_id: str,
        prompt: str,
        response_excerpt: str,
        rubric_variant: str | None = None,
        mode: str = "instant",
    ) -> JudgeResult:
        spec = resolve_rubric_spec(family_id, mode=mode)
        rubric_name = str(spec["rubric_name"])
        dimensions = self._dimension_scores(
            family_id=family_id,
            prompt=prompt,
            response_excerpt=response_excerpt,
            rubric_variant=rubric_variant,
            mode=mode,
        )
        signal = sum(dimensions.values()) / max(1, len(dimensions))
        effective_rubric_name = f"{rubric_name}:{self.rubric_version}"
        if rubric_variant:
            effective_rubric_name = f"{effective_rubric_name}:{rubric_variant}"
        return JudgeResult(
            model=self.model,
            rubric_name=effective_rubric_name,
            score=signal,
            rationale=(
                f"Deterministic fallback judge for {family_id} with dimensions "
                f"{', '.join(sorted(dimensions))} under {effective_rubric_name}."
            ),
            latency_seconds=0.01,
            dimension_scores=dimensions,
            usage={},
            metadata={
                "family_id": family_id,
                "rubric_version": self.rubric_version,
                "rubric_variant": rubric_variant,
                "mode": spec.get("active_mode"),
                "dimensions": dimensions,
                "provider_used": False,
                "judge_weight": float(spec["judge_weight"]),
            },
        )

    def _dimension_scores(
        self,
        *,
        family_id: str,
        prompt: str,
        response_excerpt: str,
        rubric_variant: str | None = None,
        mode: str = "instant",
    ) -> dict[str, float]:
        spec = resolve_rubric_spec(family_id, mode=mode)
        dimensions = [str(item) for item in spec.get("dimensions", [])]
        normalized = " ".join(response_excerpt.lower().split())
        family_id = str(family_id).strip()
        if not normalized:
            if rubric_variant:
                return {rubric_variant: 0.0}
            return {dimension: 0.0 for dimension in dimensions}
        if family_id != "general_chat":
            # No other family is wired up after the clean-slate refactor.
            raise ValueError(f"unsupported family_id for deterministic fallback: {family_id!r}")

        prompt_terms = {
            term
            for term in prompt.lower().replace(",", " ").split()
            if len(term) > 4
        }
        response_terms = set(normalized.replace(",", " ").split())
        overlap = (
            len(prompt_terms & response_terms) / max(1, len(prompt_terms))
            if prompt_terms
            else 0.0
        )
        word_count = len(normalized.split())

        citation_count = _count_any(normalized, _CITATION_MARKERS)
        evidence_signal = 1.0 if _contains_any(normalized, _EVIDENCE_MARKERS) else 0.0
        contradiction_signal = (
            1.0 if _contains_any(normalized, _CONTRADICTION_MARKERS) else 0.0
        )
        commitment_signal = 1.0 if _contains_any(normalized, _COMMITMENT_MARKERS) else 0.0
        hedging_signal = 1.0 if _contains_any(normalized, _HEDGING_MARKERS) else 0.0
        preamble_signal = 1.0 if _contains_any(normalized, _PREAMBLE_MARKERS) else 0.0

        # goal_fulfillment — how much of the prompt's content vocabulary is
        # reflected back in the response. Boosted by commitment markers (an
        # answer that takes a position usually addresses the goal).
        goal_signal = min(1.0, overlap * 1.5)
        if commitment_signal:
            goal_signal = min(1.0, goal_signal + 0.15)

        # correctness — citation/evidence presence raises confidence; visible
        # contradiction markers ("scratch that", "actually I was wrong") and
        # heavy hedging without commitment lower it.
        correctness_signal = 0.40 + (0.25 * min(1.0, citation_count / 2.0)) + (0.20 * evidence_signal)
        if contradiction_signal:
            correctness_signal -= 0.30
        if hedging_signal and not commitment_signal:
            correctness_signal -= 0.10
        correctness_signal = max(0.0, min(1.0, correctness_signal))

        # grounding — citations per supported claim. Reward raw citation count
        # up to a cap; require at least one evidence marker for full credit.
        citation_density = min(1.0, citation_count / 3.0)
        grounding_signal = 0.30 + (0.45 * citation_density) + (0.20 * evidence_signal)
        if citation_count == 0:
            grounding_signal = min(grounding_signal, 0.30)
        grounding_signal = max(0.0, min(1.0, grounding_signal))

        # conversation_coherence — penalise preamble; reward mode-appropriate
        # length. Instant mode caps reward for long responses; thinking mode
        # caps reward for very short responses.
        coherence_signal = 0.70
        if preamble_signal:
            coherence_signal -= 0.25
        active_mode = str(spec.get("active_mode") or mode or "instant").lower()
        if active_mode == "instant":
            if word_count > 250:
                coherence_signal -= 0.20
            elif word_count <= 150:
                coherence_signal += 0.15
        else:  # thinking
            if word_count < 120:
                coherence_signal -= 0.20
            elif word_count >= 250:
                coherence_signal += 0.15
        coherence_signal = max(0.0, min(1.0, coherence_signal))

        scores: dict[str, float] = {
            "goal_fulfillment": round(goal_signal, 4),
            "correctness": round(correctness_signal, 4),
            "grounding": round(grounding_signal, 4),
            "conversation_coherence": round(coherence_signal, 4),
        }
        if rubric_variant:
            return {rubric_variant: scores.get(rubric_variant, 0.0)}
        # Preserve catalog dimension order for downstream consumers.
        ordered = {dim: scores[dim] for dim in dimensions if dim in scores}
        # Fall back to scores for any catalog dimension we don't recognise.
        for dim in dimensions:
            if dim not in ordered:
                ordered[dim] = 0.0
        return ordered or scores
