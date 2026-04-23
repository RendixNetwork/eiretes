from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx
from pydantic import ValidationError

from eiretes.judge.catalog import resolve_rubric_spec
from eiretes.models import (
    VERDICT_SCORES,
    AgreementJudgeResult,
    ProviderAgreementResponse,
)
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


class LLMJudgeClient:
    """Outcome-only agreement judge client.

    Compares a candidate agent's final answer against the OpenAI baseline
    reference answer and returns a single verdict + scalar score.

    Key design choices:
      - No citations are shown to the judge. The caller is expected to have
        already stripped them from response_a / response_b before calling.
      - Position-bias mitigation still happens via the `swap` flag: when set,
        the LLM sees B then A internally; the returned verdict is un-swapped
        to caller space (A is always the candidate).
      - Refusals are treated as answers: matching refusals → matches,
        asymmetric refusals → contradicts.
    """

    def __init__(
        self,
        *,
        model: str,
        rubric_version: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
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
        self.max_tokens = int(os.getenv("EIREL_JUDGE_MAX_TOKENS", "4096"))
        self.transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(transport=self.transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def judge_agreement(
        self,
        *,
        family_id: str,
        prompt: str,
        response_a: str,
        response_b: str,
        task_mode: str | None = None,
        task_category: str | None = None,
        swap: bool = False,
    ) -> AgreementJudgeResult:
        """Judge whether response_a (candidate) agrees with response_b (baseline).

        When swap=True, the LLM internally sees B-then-A; the returned verdict
        is un-swapped so the caller's A is always the candidate agent.
        """
        if not (self.base_url and self.api_key):
            raise RuntimeError(
                "agreement judge requires EIREL_JUDGE_BASE_URL + EIREL_JUDGE_API_KEY; "
                "no deterministic fallback is provided"
            )
        spec = resolve_rubric_spec(family_id)
        rubric_name = str(spec["rubric_name"])
        valid_verdicts = tuple(spec.get("verdicts") or ())
        # Per-category prompt addenda (safety_adversarial / coding / etc.).
        # Falls back to the base system_prompt when the category is unknown
        # or not provided.
        build = spec.get("build_system_prompt")
        if callable(build):
            system_prompt = str(build(task_category))
        else:
            system_prompt = str(spec["system_prompt"])

        shown_a, shown_b = (response_b, response_a) if swap else (response_a, response_b)

        user_prompt = json.dumps(
            {
                "family_id": family_id,
                "task_mode": task_mode,
                "task_category": task_category,
                "prompt": prompt,
                "candidate_answer": shown_a,
                "reference_answer": shown_b,
                "instructions": (
                    "Return strict JSON with keys: verdict, rationale. "
                    f"verdict must be one of {list(valid_verdicts)}. "
                    "rationale should be concise and cite specific points of "
                    "agreement or divergence. Do not discuss citations, "
                    "sources, URLs, or search behavior — the candidate and "
                    "the reference use different search engines and cite "
                    "different URLs by design. Judge final-answer agreement "
                    "only."
                ),
            },
            sort_keys=True,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        started = time.perf_counter()
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        parsed = response.json()
        latency = max(0.0, time.perf_counter() - started)

        choice = (parsed.get("choices") or [{}])[0] or {}
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
                f"agreement judge returned malformed JSON (model={self.model}): {exc}"
            ) from exc
        try:
            judged = ProviderAgreementResponse.model_validate(judged_raw)
        except ValidationError as exc:
            raise ValueError(
                f"agreement judge returned invalid schema (model={self.model}): {exc}"
            ) from exc

        # Swap un-randomization: with swap=True, the LLM saw B-then-A. The
        # `matches` / `partially_matches` / `contradicts` verdicts are
        # symmetric with respect to A and B (they describe the relationship,
        # not a directional preference), so no verdict flipping is needed.
        # `not_applicable` is also symmetric. Hence no mapping is applied.
        # We only record that swap was used for auditability.
        verdict = judged.verdict
        agreement_score = VERDICT_SCORES[verdict]

        effective_rubric_name = f"{rubric_name}:{self.rubric_version}"
        return AgreementJudgeResult(
            model=self.model,
            rubric_name=effective_rubric_name,
            verdict=verdict,
            agreement_score=agreement_score,
            rationale=judged.rationale or "Agreement judge.",
            latency_seconds=latency,
            swap_applied=swap,
            usage=dict(parsed.get("usage") or {}),
            metadata={
                "family_id": family_id,
                "rubric_version": self.rubric_version,
                "task_mode": task_mode,
                "task_category": task_category,
            },
        )
