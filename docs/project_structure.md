# Project structure

```bash
genai-data-analysis/
├── .env.example              # Credential variable names (copy to .env, git-ignored)
├── .python-version           # Python version used by uv
├── pyproject.toml            # Metadata, dependencies, ruff/pytest/mypy config
├── uv.lock                   # Locked dependency versions
├── data/                     # raw/ processed/ external/ (git-ignored contents)
├── docs/                     # Documentation
├── notebooks/                # Lab: one numbered notebook per technique/experiment
├── pipelines/                # Task automation and data processing pipelines
├── reports/                  # Analysis results
├── secrets/                  # Service-account keys (git-ignored)
├── src/genaianalysis/        # Reusable package
│   ├── schema.py             # Message / Conversation / Task + jsonl save/load
│   ├── run.py                # run_task(task, convs, model) with pydantic-ai; shared cached driver
│   ├── eval.py               # score(results, convs, field): accuracy, F1, kappa, coverage, cost, ECE
│   ├── ingest/               # tabular (CSV/JSONL/parquet), hf (Hub parquet), synthetic (LLM-generated chats)
│   ├── tasks/                # Task definitions (sentiment.py)
│   ├── backends/jev.py       # TypeSafe Jev System One backend (same Task, same row shape)
│   ├── generate/             # to_migrate.py: legacy code, to be ported (Phase 1)
│   └── utils/paths.py        # Project-relative path helpers
└── tests/                    # pytest tests
```
