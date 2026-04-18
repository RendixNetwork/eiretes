"""Shared utility functions for the eiretes evaluation framework."""

from __future__ import annotations

import logging
import os
from typing import Any

_logger = logging.getLogger(__name__)


def safe_dict(value: Any) -> dict[str, Any]:
    """Return value if it's a dict, otherwise return an empty dict."""
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    """Return value if it's a list, otherwise return an empty list."""
    return value if isinstance(value, list) else []


def float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        _logger.warning("invalid float env %s=%r, using default %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        _logger.warning("env %s=%s below minimum %s, clamping", name, value, minimum)
        value = minimum
    if maximum is not None and value > maximum:
        _logger.warning("env %s=%s above maximum %s, clamping", name, value, maximum)
        value = maximum
    return value


def int_env(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        _logger.warning("invalid int env %s=%r, using default %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        _logger.warning("env %s=%s below minimum %s, clamping", name, value, minimum)
        value = minimum
    if maximum is not None and value > maximum:
        _logger.warning("env %s=%s above maximum %s, clamping", name, value, maximum)
        value = maximum
    return value
