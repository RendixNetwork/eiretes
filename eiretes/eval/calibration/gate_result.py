"""Shared gate-result type for eiretes calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


GateStatus = Literal["pass", "fail", "marginal"]


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    measured_rate: float
    threshold: float
    n_samples: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"
