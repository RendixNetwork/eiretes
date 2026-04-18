from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class JudgeResult(BaseModel):
    model: str
    rubric_name: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    latency_seconds: float = Field(ge=0.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    constraint_flags: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderJudgeResponse(BaseModel):
    """Strict validation of the JSON body returned by an LLM judge provider."""

    overall_score: float = Field(ge=0.0, le=1.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    constraint_flags: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("dimension_scores", mode="before")
    @classmethod
    def _coerce_dimension_scores(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("dimension_scores must be a JSON object")
        coerced: dict[str, float] = {}
        for key, raw in value.items():
            try:
                score = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"dimension_scores[{key!r}] must be numeric") from exc
            coerced[str(key)] = max(0.0, min(1.0, score))
        return coerced

    @field_validator("constraint_flags", mode="before")
    @classmethod
    def _coerce_constraint_flags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("constraint_flags must be a JSON array")
        return [str(item).strip() for item in value if str(item).strip()]
