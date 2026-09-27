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
│   ├── ingest/               # whatsapp (txt/zip exports), tabular, hf (Hub parquet), synthetic chats
│   ├── privacy.py            # anonymize(): pseudonyms + redaction before anything leaves the machine
│   ├── media.py              # describe_media(): attachments -> text (transcribe/describe), cached
│   ├── topics.py             # embed() -> cluster() -> label_clusters(): unsupervised topics
│   ├── tasks/                # Task definitions (sentiment.py, topics.py)
│   ├── backends/jev.py       # TypeSafe Jev System One backend (same Task, same row shape)
│   └── utils/paths.py        # Project-relative path helpers
└── tests/                    # pytest tests (+ fixtures/ with synthetic WhatsApp exports)
```
