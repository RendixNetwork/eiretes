# Eiretes

Reference-based LLM judge microservice for the **EIREL Bittensor
subnet**. Three judge roles plus a pure-function composite scorer,
all backed by a single Chutes-hosted GLM-5.1-TEE deployment.

Eiretes is consumed as an HTTP sidecar by `eirel-ai`'s validator
engine. It is **stateless** — no parameter pool, no oracle calls, no
caching. The validator passes structured `JudgeInputBundle`s in and
gets per-task verdicts out. Aggregation, weighting, and
chain-publication live in `eirel-ai`.

## Layout

```
eiretes/
├── eiretes/
│   ├── service.py             # FastAPI app — /healthz + 4 judge endpoints
│   ├── utils.py               # safe_dict / safe_list / float_env / int_env
│   └── eval/
│       ├── bundle.py          # JudgeInputBundle (Pydantic) — role × budget dispatch
│       ├── judge.py           # EvalJudge — reference-based outcome + guidance
│       ├── multi_judge.py     # MultiJudge — outer dimensions in one call
│       ├── pairwise.py        # PairwiseJudge — A/B preference (with optional expected_answer anchor)
│       ├── composite.py       # composite_score — pure-function multiplicative scorer
│       ├── safety_attestation.py  # regex denylist + chat-template token-leak detection
│       ├── feedback.py        # EvalFeedbackDoc assembly (test-only in-memory store)
│       ├── models.py          # Shared Pydantic types
│       ├── providers/         # OpenAI-compatible client (used by judges)
│       ├── calibration/       # Calibration helpers (offline)
│       └── config.py          # JudgeConfig — resolves EIREL_EVAL_JUDGE_* envs
└── tests/
```

## Architecture

The validator builds a `JudgeInputBundle` per task and dispatches it
to the right judge based on role:

```
                     JudgeInputBundle
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  PairwiseJudge        MultiJudge          EvalJudge
  (A vs B preference)  (grounded /         (reference-based outcome
                       retrieval /          + categorical guidance)
                       safety in
                       one call)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    composite_score (pure)
                            ▼
                   per-task multiplicative
                   composite with hard gates
```

All three LLM judges share a single Chutes-hosted GLM-5.1-TEE
deployment — the only thing that varies per role is the prompt /
schema, which lives inside each judge module.

## HTTP API

### `GET /healthz`

```json
{"status": "ok", "judge_model": "zai-org/GLM-5.1-TEE", "rubric_version": "eval_judge_v1"}
```

### `POST /v1/judge/eval`

Reference-based LLM-as-judge. The validator passes a structured
bundle including `expected_claims` + `must_not_claim` derived from the
3-oracle reconciliation; the judge returns an `EvalOutcome`
(`correct` / `partial` / `wrong` / `hallucinated` / `refused` /
`disputed`) plus a categorical `failure_mode` and one-sentence
guidance.

### `POST /v1/judge/multi`

One-call outer-metric judge. Scores `grounded_correctness`,
`retrieval_quality`, `instruction_safety` in a single LLM call.
Pre-extracted `expected_claims` go into `bundle.constraints`;
`candidate_citations` are passed alongside.

### `POST /v1/judge/pairwise`

Pairwise preference (A vs B) with position-bias defense delegated to
the caller (call twice with A/B swapped, average the two scores). When
`expected_answer` is supplied, the judge anchors preference on factual
agreement — closes the calibration quirk where verbose miner answers
always beat the terse oracle reference.

### `POST /v1/judge/eval/composite`

**Pure function — no LLM call.** Combines an `EvalOutcome` with the
per-dimension scores from MultiJudge plus server-attested factors
(tool ledger, cost) into the multiplicative composite:

```
composite = clip(
    grounded_gate × safety_gate × safety_attestation
    × tool_attestation × efficiency × hallucination
    × cost_attestation × (outcome_score + pairwise_bonus),
    0, 1
)
```

Hard gates: `grounded_correctness ≥ 0.60`, `instruction_safety ≥
0.80`. Pairwise contributes a `±0.10` bonus on top of the outcome
score.

Per-miner feedback retrieval lives on owner-api directly
(hotkey-signed `GET /v1/eval/feedback`) — eiretes is purely the judge
service.

## Configuration

Single env block — all three judge roles read from the same set:

| name | default | notes |
|------|---------|-------|
| `EIREL_EVAL_JUDGE_BASE_URL` | — | OpenAI-compatible chat completion endpoint (Chutes default: `https://llm.chutes.ai/v1`) |
| `EIREL_EVAL_JUDGE_API_KEY` | — | bearer token; **funds the judge LLM bill** |
| `EIREL_EVAL_JUDGE_MODEL` | `local-rubric-judge` | Chutes-hosted model name (recommended: `zai-org/GLM-5.1-TEE`) |
| `EIREL_EVAL_JUDGE_TIMEOUT_SECONDS` | `30` | per-request HTTP timeout |
| `EIREL_EVAL_JUDGE_MAX_TOKENS` | `2048` | per-request output cap; reasoning models commonly need 4096+ |
| `EIREL_EVAL_MIN_TURN_COST_USD` | `0.00005` | cost-attestation knockout floor |
| `EIRETES_JUDGE_PORT` | `8095` | HTTP bind port |

Non-numeric values are logged and replaced with the default — bad env
vars will not crash startup.

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]

# unit tests
python -m pytest

# start the service
eiretes-judge-service
# or: uvicorn eiretes.service:app --host 0.0.0.0 --port 8095
```

In the validator stack, this is the `eiretes-judge` sidecar; see
`eirel-ai/docker-compose.validator.yml`.

## Python version

Requires Python >= 3.12. Every module uses `from __future__ import annotations`.
