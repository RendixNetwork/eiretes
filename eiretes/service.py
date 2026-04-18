from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from eiretes.judge.catalog import RUBRIC_CATALOG
from eiretes.judge.llm_judge import LLMJudgeClient
from eiretes.models import JudgeResult
from eiretes.utils import float_env, int_env

_logger = logging.getLogger(__name__)


# -- Request / Response models -----------

_MAX_PROMPT_CHARS = 32_000
_MAX_EXCERPT_CHARS = 200_000

ChatMode = Literal["instant", "thinking"]


class JudgeRequest(BaseModel):
    family_id: str = Field(min_length=1, max_length=64)
    prompt: str = Field(max_length=_MAX_PROMPT_CHARS)
    response_excerpt: str = Field(max_length=_MAX_EXCERPT_CHARS)
    rubric_variant: str | None = Field(default=None, max_length=128)
    mode: ChatMode = Field(default="instant")


# -- Application state -----------

_judge: LLMJudgeClient | None = None


def _build_judge() -> LLMJudgeClient:
    return LLMJudgeClient(
        model=os.getenv("EIREL_JUDGE_MODEL", "local-rubric-judge"),
        rubric_version=os.getenv("EIREL_JUDGE_RUBRIC_VERSION", "general_chat_rubric_v1"),
        base_url=os.getenv("EIREL_JUDGE_BASE_URL") or None,
        api_key=os.getenv("EIREL_JUDGE_API_KEY"),
        timeout_seconds=float_env("EIREL_JUDGE_TIMEOUT_SECONDS", 30.0, minimum=0.1),
        ensemble_base_url=os.getenv("EIREL_ENSEMBLE_JUDGE_BASE_URL") or None,
        ensemble_api_key=os.getenv("EIREL_ENSEMBLE_JUDGE_API_KEY") or None,
        ensemble_timeout_seconds=float_env(
            "EIREL_ENSEMBLE_JUDGE_TIMEOUT_SECONDS", 30.0, minimum=0.1
        ),
        ensemble_disagreement_threshold=float_env(
            "EIREL_ENSEMBLE_JUDGE_DISAGREEMENT_THRESHOLD", 0.20, minimum=0.0, maximum=1.0
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _judge
    _judge = _build_judge()
    _logger.info(
        "judge service ready: model=%s rubric=%s",
        _judge.model,
        _judge.rubric_version,
    )
    try:
        yield
    finally:
        if _judge is not None:
            await _judge.aclose()
        _judge = None


app = FastAPI(title="Eiretes Judge Service", lifespan=lifespan)


def get_judge() -> LLMJudgeClient:
    if _judge is None:
        raise HTTPException(status_code=503, detail="judge not initialized")
    return _judge


def _require_known_family(family_id: str) -> None:
    if family_id not in RUBRIC_CATALOG:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_family_id",
                "family_id": family_id,
                "valid_families": sorted(RUBRIC_CATALOG),
            },
        )


# -- Endpoints -----------

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "judge_model": _judge.model if _judge else None,
        "rubric_version": _judge.rubric_version if _judge else None,
    }


@app.get("/v1/catalog")
def catalog(
    judge_client: LLMJudgeClient = Depends(get_judge),
) -> dict[str, Any]:
    """Publish the rubric catalog so consumers don't hold a stale static snapshot.

    Only the fields needed by eirel-ai's dispatcher are exposed — system prompts
    and per-dimension rubric anchors stay server-side so miners can't mine them.
    """
    families: dict[str, Any] = {}
    for family_id, spec in RUBRIC_CATALOG.items():
        supported_modes = sorted((spec.get("system_prompt_by_mode") or {}).keys())
        families[family_id] = {
            "rubric_name": spec.get("rubric_name"),
            "judge_mode": spec.get("judge_mode"),
            "judge_weight": spec.get("judge_weight"),
            "dimensions": list(spec.get("dimensions") or []),
            "supported_modes": supported_modes,
        }
    return {
        "rubric_version": judge_client.rubric_version,
        "judge_model": judge_client.model,
        "families": families,
    }


@app.post("/v1/judge")
async def judge(
    req: JudgeRequest,
    judge_client: LLMJudgeClient = Depends(get_judge),
) -> JudgeResult:
    _require_known_family(req.family_id)
    return await judge_client.judge(
        family_id=req.family_id,
        prompt=req.prompt,
        response_excerpt=req.response_excerpt,
        rubric_variant=req.rubric_variant,
        mode=req.mode,
    )


# -- Entry point -----------

def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int_env("EIRETES_JUDGE_PORT", 8095, minimum=1, maximum=65535),
    )


if __name__ == "__main__":
    main()
