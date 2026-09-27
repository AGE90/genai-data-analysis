# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Managed with uv (Python pinned in `.python-version`); `uv sync` installs the package in editable mode plus the dev group.

```bash
uv sync
uv run pytest                                   # all tests
uv run pytest tests/test_paths.py::test_paths_resolve_from_project_root   # single test
uv run ruff check .        # lint (ruff format . to format)
uv run mypy src
uv run jupyter nbconvert --execute --to notebook --output-dir <tmp> notebooks/<nb>.ipynb   # run a notebook headless
uv add <pkg> / uv add --dev <pkg>
```

## What this repo is

A data-science toolbox + lab for turning human data (WhatsApp chats, surveys, tickets, audio/images) into typed, evaluated results with LLMs and other generative models. Notebooks in `notebooks/` are the lab; code reused across notebooks gets promoted into `src/genaianalysis/`. Work happens on `dev`.

## Architecture (target design, being built in phases)

Phase 0 (uv, env auth) and phases 2–3 (benchmark, Jev) are built; phase 1 is only partly done (WhatsApp parser and `privacy.py` are still missing). The code rests on three stable seams; everything else should stay replaceable:

1. **Canonical data model**: every source (WhatsApp export `.txt`, CSV/JSONL, HF datasets, synthetic LLM-generated chats) is normalized into `Conversation[Message]` (text + optional media) in `schema.py`. A new source is one loader function in `ingest/`.
2. **Task = pydantic output schema + instructions** (`tasks/`), e.g. a `SentimentArc` with `Literal` labels. Use typed outputs, never manual JSON parsing.
3. **Model = a pydantic-ai model string** (`"google:gemini-flash-latest"`, `"anthropic:claude-sonnet-5"`, `"ollama:…"`). Non-LLM backends (TypeSafe's Jev, local classifiers, embeddings) are plain functions with the same `(task, conversations) -> DataFrame` shape. Jev answers typed Choice/Score/Noul questions with probabilities + confidence; map `Literal` fields to Choice, ordinal fields to Score, and bool fields to Noul.

`run.cached_rows` is the shared driver for every backend. It gives bounded concurrency, `rpm` pacing, a per-item disk cache in `data/processed/cache/<backend>/`, and failures kept as rows with an `error` column (a failed item never raises). All backends return the same row shape: `id, task, backend, model, resolved_model, <output fields>, latency_s, input_tokens, output_tokens, cost_usd, error`; Jev adds `<field>_confidence` and `<field>_proba`.

Plus `eval.score` (accuracy / macro-F1 / kappa / coverage / latency / cost / ECE vs gold in `Conversation.labels[field]`) so models are compared on the same data, a planned `privacy.py` redaction step (pseudonymize senders, strip phones/emails/IDs) before data is sent to any API, and results logged in `docs/research.md`. Don't add registries, factories or plugin systems until a second real need exists.

Planned phases: 1 = core pipeline with parity to the legacy WhatsApp sentiment system; 2 = HF + synthetic datasets and a model benchmark; 3 = Jev vs LLM (accuracy, calibration, latency, cost); 4+ = one notebook per technique (topic clustering, audio transcription, LLM-as-judge, distillation).

## Legacy code

`src/genaianalysis/generate/to_migrate.py` is a pasted dump of two modules from a former employer's system (Firestore/GCS/BigQuery loaders + Vertex Gemini media-to-text and sentiment). **It is not valid Python** and is excluded from ruff and mypy in `pyproject.toml`. Port its ideas (`extract_media_from_json`, conversation formatting, media prompts, the sentiment prompt/examples), drop the company-specific infrastructure, then delete the file.

## Credentials & models

- There is no credentials module. SDKs read standard env vars from `.env` (names in `.env.example`): `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, and for Vertex `GOOGLE_APPLICATION_CREDENTIALS` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`. Key files go in `secrets/` (git-ignored).
- Load `.env` with an explicit path: `load_dotenv(project_dir(".env"))`. A bare `load_dotenv()` fails when run from stdin/`-c`.
- pydantic-ai prefixes: `google:` (Gemini API), `anthropic:`, `openai:`, `ollama:`. `Agent.run` returns a result whose `.usage` is a property (not a method); `.usage.cost` is the USD cost from genai-prices.
- Gemini free tier allows 15 requests/min per model (and `gemini-flash-latest` → gemini-3.8-flash only **20 requests/day**, so free-tier runs use `gemini-flash-lite-latest`), so pass `rpm=12` to `run_task`/`generate_chats`. `with_retries` backs off for ~62 s total to survive 429/503 errors.
- Jev: package `typesafe-sdk` (import `typesafe_sdk`), env `TYPESAFE_API_KEY`, `client.system_one(state, {name: Choice|Noul|Score})`. The SDK retries on its own. Text only, strongest in English. Docs as markdown: `https://docs.typesafe.ai/<page>.md` (index at `/llms.txt`).
- Gemini model ids get retired quickly (`gemini-2.0-flash` and `gemini-2.5-flash` already return 404), so default to aliases like `gemini-flash-latest`. List live ids with `genai.Client().models.list()`. 503 "high demand" errors are transient on Google's side.
- `utils/paths.py` gives project-root-relative helpers (`project_dir`, `data_raw_dir`, …) via pyprojroot; use them instead of relative paths.
