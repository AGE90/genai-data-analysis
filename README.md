# Gen AI Data Analysis

A data-science toolbox and lab for turning human data (chats, surveys, tickets, audio, images) into structured, evaluated results using LLMs and other generative models.

## Overview

- Any source is normalized into one conversation format, any task is a typed (pydantic) output schema, and any model is a name you can swap (Gemini, Claude, OpenAI, local, System-One models such as TypeSafe's Jev).
- Every technique is compared on the same labeled data, so upgrades are judged by numbers.
- Notebooks are the lab; code reused across notebooks moves into `src/genaianalysis`.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AGE90/genai-data-analysis.git
    cd genai-data-analysis
    ```

2. Follow the installation guide in the [docs/install.md](docs/install.md) file.

## Project Structure

A detailed project structure is available in the [docs/project_structure.md](docs/project_structure.md) file.

## Development

Managed with [uv](https://docs.astral.sh/uv/): `uv sync`, then `uv run pytest`, `uv run ruff check .`, `uv run mypy src`.
