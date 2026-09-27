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
│   ├── data/                 # Data loading / processing
│   ├── generate/             # Generative-model tasks (to_migrate.py: legacy code, ported in Phase 1)
│   └── utils/paths.py        # Project-relative path helpers
└── tests/                    # pytest tests
```
