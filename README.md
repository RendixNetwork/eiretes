# Eiretes

LLM judge microservice for the **EIREL Bittensor subnet**. Scores miner
responses against versioned family rubrics.

Eiretes is consumed as an HTTP sidecar by `eirel-ai` (validator engine,
benchmark orchestrator). Family-specific weighting, aggregation, and
anti-gaming live in `eirel-ai`; this service owns **rubric definitions**
and **LLM-backed evaluation** only.

## Layout

```
eiretes/
├── eiretes/
│   ├── service.py          # FastAPI app — /healthz, /v1/catalog, /v1/judge
│   ├── models.py           # JudgeResult + ProviderJudgeResponse pydantic models
│   ├── utils.py            # safe_dict / safe_list / float_env / int_env
│   └── judge/
│       ├── catalog.py      # RUBRIC_CATALOG — general_chat quality rubric
│       ├── llm_judge.py    # LLMJudgeClient — ensemble + deterministic fallback
│       └── rubrics/
│           └── general_chat.py  # Dimension weights, mode prompts, 1–4 anchors
└── tests/
    ├── test_judge_service.py
    └── test_ensemble_judge.py
```

## Rubric catalog

`RUBRIC_CATALOG` in `eiretes/judge/catalog.py` is the single source of
truth for how each execution family is scored. Only one launch family
is active:

| family         | modes                | dimensions                                                       |
|----------------|----------------------|------------------------------------------------------------------|
| `general_chat` | `instant`, `thinking`| `goal_fulfillment` (0.45), `correctness` (0.25), `grounding` (0.15), `conversation_coherence` (0.15) |

Every dimension uses a 1–4 Likert anchor mapped to 0.25 / 0.5 / 0.75 /
1.0. The active system prompt is selected per request by the `mode`
field (`instant` or `thinking`) — `resolve_rubric_spec` copies the
catalog entry and sets `active_mode` / `active_system_prompt` so the
judge client can stream the right instructions without mutating shared
state.

Future families (`deep_research`, `coding`) plug in by adding an entry
to `RUBRIC_CATALOG` plus a rubric module under
`eiretes/judge/rubrics/`.

## HTTP API

### `GET /healthz`

```json
{"status": "ok", "judge_model": "...", "rubric_version": "general_chat_rubric_v1"}
```

### `GET /v1/catalog`

Publishes the rubric catalog so consumers don't hold a stale static
snapshot. Exposes only what `eirel-ai`'s dispatcher needs — system
prompts and per-dimension anchor text stay server-side so miners can't
mine them:

```json
{
  "rubric_version": "general_chat_rubric_v1",
  "judge_model": "...",
  "families": {
    "general_chat": {
      "rubric_name": "general_chat_quality_rubric_v1",
      "judge_mode": "judge_primary",
      "judge_weight": 1.0,
      "dimensions": ["goal_fulfillment", "correctness", "grounding", "conversation_coherence"],
      "supported_modes": ["instant", "thinking"]
    }
  }
}
```

### `POST /v1/judge`

Request:
```json
{
  "family_id": "general_chat",
  "prompt": "Summarize Q1 revenue drivers",
  "response_excerpt": "...",
  "mode": "instant",
  "rubric_variant": null
}
```

Returns a `JudgeResult` (see `eiretes/models.py`): `model`,
`rubric_name`, `score`, `rationale`, `dimension_scores`,
`constraint_flags`, `latency_seconds`, `usage`, `metadata`. Invalid
`family_id` → HTTP 400 with the valid family list.

## Judge backend

`LLMJudgeClient` talks to any OpenAI-compatible chat completion endpoint
via `EIREL_JUDGE_BASE_URL` / `EIREL_JUDGE_API_KEY`. When a second
endpoint is configured (`EIREL_ENSEMBLE_*`), primary and ensemble judges
run and scores are averaged; disagreement above
`EIREL_ENSEMBLE_JUDGE_DISAGREEMENT_THRESHOLD` surfaces as a
`judge_disagreement:<delta>` constraint flag.

If no provider is configured **or** the provider call fails, the client
falls back to a deterministic token-based judge so evaluation never
blocks.

## Environment variables

| name | default | notes |
|------|---------|-------|
| `EIREL_JUDGE_MODEL` | `local-rubric-judge` | model name passed to the provider |
| `EIREL_JUDGE_RUBRIC_VERSION` | `general_chat_rubric_v1` | stamped into `rubric_name` + `metadata` |
| `EIREL_JUDGE_BASE_URL` | — | OpenAI-compatible chat completion endpoint |
| `EIREL_JUDGE_API_KEY` | — | bearer token for the primary judge |
| `EIREL_JUDGE_TIMEOUT_SECONDS` | `30` | primary provider timeout |
| `EIREL_ENSEMBLE_JUDGE_BASE_URL` | — | enables ensemble mode when set |
| `EIREL_ENSEMBLE_JUDGE_API_KEY` | — | bearer token for the secondary judge |
| `EIREL_ENSEMBLE_JUDGE_TIMEOUT_SECONDS` | `30` | secondary provider timeout |
| `EIREL_ENSEMBLE_JUDGE_DISAGREEMENT_THRESHOLD` | `0.20` | flag threshold (0-1) |
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

In the `eirel-ai` docker stack, this is the `eiretes-judge` sidecar; see
`eirel-ai/docker-compose.yml`.

## Python version

Requires Python >= 3.12. Every module uses `from __future__ import annotations`.
