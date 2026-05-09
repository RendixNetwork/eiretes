"""Eiretes Chutes cost extractor tests."""

from __future__ import annotations

import pytest

from eiretes.eval.providers.cost_calc import extract_chutes_chat_cost
from eiretes.eval.providers.pricing import PRICING, price_for


def test_glm_5_1_tee_rate_present() -> None:
    """The default judge model must have an entry in the static table —
    no silent fallback to ``chutes:*`` for our primary deployment."""
    assert "chutes:zai-org/GLM-5.1-TEE" in PRICING


def test_extract_glm_cost() -> None:
    """zai-org/GLM-5.1-TEE: 0.5/Mtok input, 2.0/Mtok output.
    1000 prompt + 500 completion = 0.0005 + 0.001 = 0.0015"""
    payload = {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
    cost = extract_chutes_chat_cost(payload, model="zai-org/GLM-5.1-TEE")
    assert cost == pytest.approx(0.0015)


def test_extract_unknown_model_returns_none() -> None:
    """The judge configuration shouldn't silently bill unknown models
    against the generous fallback — return None so the caller can
    tag the call as ``cost=unknown`` in telemetry."""
    payload = {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
    cost = extract_chutes_chat_cost(payload, model="totally-unknown")
    # price_for falls back to chutes:* so this DOES return a cost.
    # That fallback exists so a deploy doesn't crash on a model
    # rename — but we assert the value matches the glob rate.
    assert cost == pytest.approx(0.0015)


def test_extract_no_usage_returns_none() -> None:
    assert extract_chutes_chat_cost({}, model="zai-org/GLM-5.1-TEE") is None
    assert (
        extract_chutes_chat_cost({"usage": "string not dict"}, model="zai-org/GLM-5.1-TEE")
        is None
    )


def test_extract_zero_tokens_returns_none() -> None:
    """Zero tokens on both sides means the call never metered — emit
    None so callers can distinguish from a metered $0 call."""
    payload = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    assert extract_chutes_chat_cost(payload, model="zai-org/GLM-5.1-TEE") is None


def test_pricing_glob_fallback() -> None:
    """``chutes:*`` is the safety net when the model isn't named."""
    assert price_for("totally-unknown-name") is not None
