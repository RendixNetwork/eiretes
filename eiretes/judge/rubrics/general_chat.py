from __future__ import annotations

from typing import Any


# -- Mode-specific judge instructions -----------------------------------------
# The same dimensions are scored in both modes; the system prompt nudges the
# judge to weight depth/conciseness expectations differently per mode. The
# anchor language inside `dimension_rubrics` is mode-agnostic so the judge can
# apply the same scoring scale regardless of mode.

_INSTANT_SYSTEM_PROMPT = (
    "You are the official EIREL general_chat judge for INSTANT mode. "
    "Instant mode targets a tight latency budget and short output. "
    "Reward responses that get to the point quickly, satisfy the user's goal in "
    "a few crisp paragraphs, and avoid filler, preamble, or self-narration. "
    "Penalize verbose responses that pad with restated context, multi-section "
    "headers, or boilerplate hedging. Depth is appropriate only when the goal "
    "actually requires it; do not reward length for its own sake. "
    "Score grounding strictly on whether tool-sourced claims carry visible "
    "citations or anchors — do not rescue unsupported claims."
)

_THINKING_SYSTEM_PROMPT = (
    "You are the official EIREL general_chat judge for THINKING mode. "
    "Thinking mode permits a longer latency budget and a larger output budget "
    "so the agent can reason carefully and integrate evidence. "
    "Reward responses that decompose the question, work through evidence, "
    "address competing considerations, and deliver a substantive conclusion. "
    "Do not reward verbosity that does not advance the answer. "
    "Penalize responses that remain shallow, skip key sub-questions, or fail "
    "to use the additional budget meaningfully. "
    "Score grounding strictly on whether tool-sourced claims carry visible "
    "citations or anchors — do not rescue unsupported claims."
)


GENERAL_CHAT_QUALITY_RUBRIC: dict[str, Any] = {
    "rubric_name": "general_chat_quality_rubric_v1",
    "judge_weight": 1.0,
    "judge_mode": "judge_primary",
    "ensemble_mode": True,
    "dimensions": (
        "goal_fulfillment",
        "correctness",
        "grounding",
        "conversation_coherence",
    ),
    "weights": {
        "goal_fulfillment": 0.45,
        "correctness": 0.25,
        "grounding": 0.15,
        "conversation_coherence": 0.15,
    },
    "system_prompt_by_mode": {
        "instant": _INSTANT_SYSTEM_PROMPT,
        "thinking": _THINKING_SYSTEM_PROMPT,
    },
    "dimension_rubrics": {
        "goal_fulfillment": {
            "description": (
                "Did the user's underlying goal get satisfied by the end of the "
                "conversation? Judge the final state, not just the last turn."
            ),
            "score_5": (
                "The user's goal is fully and unambiguously satisfied. The response "
                "directly addresses the request, covers all stated sub-goals, and "
                "leaves no obvious follow-up required to act on the answer."
            ),
            "score_4": (
                "The primary goal is satisfied. Minor sub-goals or peripheral "
                "details may be missed but the user can act on the response without "
                "having to re-ask the core question."
            ),
            "score_3": (
                "The goal is partially addressed. The response is on-topic but "
                "leaves a meaningful gap, omits a key sub-question, or stops short "
                "of the deliverable the user asked for."
            ),
            "score_2": (
                "The response engages the topic but does not deliver on the goal. "
                "It substitutes adjacent information, deflects, or answers a "
                "different question than the one posed."
            ),
            "score_1": (
                "The goal is not addressed. The response is off-topic, refuses "
                "without basis, or omits the primary deliverable entirely."
            ),
            "penalties": [
                "answering a materially different question than the one posed",
                "refusing to answer without a stated, valid basis",
                "stopping the response before the requested deliverable",
                "deferring to a follow-up turn when the user asked for a final answer",
            ],
        },
        "correctness": {
            "description": (
                "Are factual claims accurate? Is reasoning sound? Is any code, "
                "math, or step-by-step logic free of errors?"
            ),
            "score_5": (
                "All factual claims are accurate; reasoning is sound and traceable; "
                "any code, math, or worked steps are correct end-to-end with no "
                "silent errors."
            ),
            "score_4": (
                "Core claims are accurate and reasoning is sound. Minor inaccuracies "
                "on peripheral details that do not change the conclusion."
            ),
            "score_3": (
                "Mostly accurate with one or more notable factual or reasoning "
                "errors. The user could be misled on a non-critical point."
            ),
            "score_2": (
                "Several errors of fact, math, or reasoning. The conclusion is "
                "partially undermined; a careful user would need to re-verify."
            ),
            "score_1": (
                "Pervasive factual or reasoning errors; fabricated facts; "
                "self-contradictory claims; broken code or logic; conclusion is "
                "unreliable."
            ),
            "penalties": [
                "fabricated facts presented confidently",
                "self-contradiction within the same response",
                "code or math that does not run / does not produce the stated result",
                "reasoning that contradicts evidence cited in the same response",
            ],
        },
        "grounding": {
            "description": (
                "For tool-sourced claims, are sources properly attributed and "
                "supported? This is a soft signal — the hard trace integrity gate "
                "is enforced upstream in eirel-ai. Here we check citation quality "
                "and whether claims line up with the apparent source."
            ),
            "score_5": (
                "Every tool-sourced claim is paired with a clear citation or "
                "anchor; sources are authoritative and directly support the claim; "
                "no orphan facts presented as retrieved knowledge."
            ),
            "score_4": (
                "Most tool-sourced claims are cited with relevant sources. "
                "Occasional missing citation on a non-critical point but no "
                "core claim is unattributed."
            ),
            "score_3": (
                "Mixed attribution. Several tool-sourced claims lack citations or "
                "the citations look weak / tangential to what they purportedly "
                "support."
            ),
            "score_2": (
                "Tool-sourced claims are largely unattributed; sources are weak, "
                "stale, or do not back the assertions made."
            ),
            "score_1": (
                "Tool-sourced claims are presented with no attribution at all, or "
                "the citations appear fabricated or do not match the cited content."
            ),
            "penalties": [
                "fabricated or hallucinated citations",
                "citing sources that do not support the stated claim",
                "presenting retrieved facts as the agent's own knowledge",
                "orphan URLs with no claim attached",
            ],
        },
        "conversation_coherence": {
            "description": (
                "Does the response follow conversational norms? It should be "
                "context-aware, mode-appropriate in depth (instant = crisp; "
                "thinking = thorough), and free of canned filler or unnecessary "
                "preamble."
            ),
            "score_5": (
                "Tone, length, and depth are perfectly matched to the mode. No "
                "filler, no preamble, no self-narration. The response reads as a "
                "direct continuation of the conversation."
            ),
            "score_4": (
                "Generally well-calibrated to the mode. Minor preamble or a single "
                "filler phrase, but the substance is delivered cleanly."
            ),
            "score_3": (
                "Noticeable mode mismatch: padded in instant mode, or thin in "
                "thinking mode. Some boilerplate or restatement of the question "
                "before answering."
            ),
            "score_2": (
                "Significant mismatch — long preamble, canned hedging, or "
                "obviously truncated depth. The user has to skim past filler to "
                "reach the answer."
            ),
            "score_1": (
                "Heavy filler, repeated boilerplate, or response shape that "
                "ignores the conversational context entirely (e.g., dumping a "
                "report when the user asked a one-line question)."
            ),
            "penalties": [
                "preamble like 'certainly', 'great question', 'I'd be happy to'",
                "restating the user's question before answering it",
                "long structured report when the user asked a casual question",
                "one-line dismissive reply when the user asked a substantive question",
                "self-narration of the agent's reasoning process when not asked",
            ],
        },
    },
}
