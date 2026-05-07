"""Eiretes-side env-var taxonomy for the eval pipeline.

Eiretes runs ONE external dependency in production: the Chutes-hosted
``zai-org/GLM-5.1-TEE`` model used by all three internal judge roles
(``pairwise``, ``multi``, ``eval``). The same model + base_url + api_key
covers every judge — the only thing that varies per role is the
prompt/schema, which lives inside each judge module.

Feedback persistence reaches into eirel-ai owner-api Postgres via
internal-token-gated HTTP. Eiretes itself stays stateless.

Eiretes does NOT host the parameter pool, does NOT call oracles, and
does NOT cache. See `project_eval_repo_separation.md` for the full
3-repo split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _strip(value: str | None) -> str:
    return (value or "").strip()


def _strip_or_none(value: str | None) -> str | None:
    cleaned = _strip(value)
    return cleaned or None


def _float_env(name: str, default: float) -> float:
    raw = _strip(os.getenv(name))
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"env {name}={raw!r} is not a valid float"
        ) from exc


def _int_env(name: str, default: int) -> int:
    raw = _strip(os.getenv(name))
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"env {name}={raw!r} is not a valid int"
        ) from exc


@dataclass(frozen=True)
class JudgeConfig:
    """Single Chutes-hosted GLM-5.1-TEE config used by all judge roles."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_tokens: int

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


_DEFAULT_JUDGE_MODEL = "zai-org/GLM-5.1-TEE"
_DEFAULT_JUDGE_TIMEOUT_SECONDS = 30.0
_DEFAULT_JUDGE_MAX_TOKENS = 2048


def judge_config() -> JudgeConfig:
    """Resolve ``EIREL_EVAL_JUDGE_*`` envs.

    Single config used by ``pairwise``, ``multi``, ``eval`` judges.
    """
    base_url = _strip(os.getenv("EIREL_EVAL_JUDGE_BASE_URL", "")).rstrip("/")
    api_key = _strip(os.getenv("EIREL_EVAL_JUDGE_API_KEY", ""))
    model = _strip(os.getenv("EIREL_EVAL_JUDGE_MODEL", "")) or _DEFAULT_JUDGE_MODEL
    timeout_seconds = _float_env(
        "EIREL_EVAL_JUDGE_TIMEOUT_SECONDS", _DEFAULT_JUDGE_TIMEOUT_SECONDS,
    )
    max_tokens = _int_env(
        "EIREL_EVAL_JUDGE_MAX_TOKENS", _DEFAULT_JUDGE_MAX_TOKENS,
    )
    return JudgeConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
    )


def min_turn_cost_usd() -> float:
    """Cost-attestation knockout floor.

    A miner whose per-turn proxy cost falls below this threshold
    triggers ``cost_attestation_knockout=0``. Default $0.00005 captures
    cached/short-circuited responses that didn't actually run inference.
    """
    return _float_env("EIREL_EVAL_MIN_TURN_COST_USD", 0.00005)
