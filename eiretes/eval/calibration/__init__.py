"""Eiretes-side calibration harness.

Two gates the operator runs at deploy time before flipping production
traffic to a new judge model (Chutes-hosted ``zai-org/GLM-5.1-TEE``):

  * **JSON parse-rate gate** — measures parseability of the structured
    response across N calls per judge role (pairwise / multi / eval).
    Acceptance: ≥98% parse rate, ≥95% schema-valid.
  * **Pairwise position-bias chi-square** — verifies the random A/B
    assignment in the validator (``secrets.randbelow(2)``) yields a
    uniform distribution. Pure local; no LLM needed.

Both gates produce ``GateResult`` objects the operator can dump as
JSON for the deploy record.
"""

from __future__ import annotations

from eiretes.eval.calibration.gate_result import GateResult, GateStatus
from eiretes.eval.calibration.json_parse_rate import (
    JudgeJsonParseRateFixture,
    JudgeJsonParseRateGate,
    measure_judge_json_parse_rate,
)
from eiretes.eval.calibration.position_bias import (
    DEFAULT_POSITION_BIAS_ALPHA,
    measure_pairwise_position_bias,
)

__all__ = [
    "DEFAULT_POSITION_BIAS_ALPHA",
    "GateResult",
    "GateStatus",
    "JudgeJsonParseRateFixture",
    "JudgeJsonParseRateGate",
    "measure_judge_json_parse_rate",
    "measure_pairwise_position_bias",
]
