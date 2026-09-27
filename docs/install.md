# Installation

Requires [uv](https://docs.astral.sh/uv/). uv installs the pinned Python (`.python-version`) and all dependencies:

```bash
git clone https://github.com/AGE90/genai-data-analysis.git
cd genai-data-analysis
uv sync                 # creates .venv, installs the package (editable) + dev tools
cp .env.example .env    # then fill in the keys you use
```

Common commands:

```bash
uv run pytest           # tests
uv run ruff check .     # lint   (ruff format . to format)
uv run mypy src         # types
uv add <pkg>            # add a dependency (uv add --dev <pkg> for dev-only)
```

## Notebooks

Select the `.venv` kernel. Notebooks load `.env` via `genaianalysis.utils.paths.project_dir(".env")` and use:

```python
%load_ext autoreload
%autoreload 2
```

## Credentials

Variable names follow each SDK's defaults, so clients pick them up without a credentials module (see `.env.example`):

- `GEMINI_API_KEY`: Google AI Studio (notebook 01, pydantic-ai `google-gla:`)
- `ANTHROPIC_API_KEY`: Claude (pydantic-ai `anthropic:`)
- `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`: Vertex AI (notebook 02). Keep key files in `secrets/` (git-ignored).
