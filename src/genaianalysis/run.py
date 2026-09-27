"""Run a Task over Conversations with any pydantic-ai model string; one row per conversation.

Row shape shared by every backend: id, task, backend, model, resolved_model, <output fields>,
latency_s, input_tokens, output_tokens, cost_usd, error.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models import Model

from genaianalysis.schema import Conversation, Task
from genaianalysis.utils.paths import data_processed_dir

logger = logging.getLogger(__name__)
T = TypeVar("T")

CACHE_DIR = data_processed_dir("cache")


async def with_retries(fn: Callable[[], Awaitable[T]], attempts: int = 6) -> T:
    """Retry transient failures (503 'high demand', 429 quota) with backoff: 2+4+8+16+32 s,
    enough to outlast a per-minute quota window."""
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001 - provider SDKs raise many unrelated types
            if i == attempts - 1:
                raise
            logger.warning("attempt %d failed (%s), retrying", i + 1, e)
            await asyncio.sleep(2 ** (i + 1))
    raise AssertionError("unreachable")


def cache_key(backend: str, task: Task, model: str, conv: Conversation) -> str:
    payload = json.dumps(
        [backend, task.name, task.instructions, task.output_type.model_json_schema(), model,
         conv.model_dump(mode="json", exclude={"labels"})],
        sort_keys=True,
    )  # fmt: skip
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


async def cached_rows(
    backend: str,
    task: Task,
    model: str,
    convs: Sequence[Conversation],
    one: Callable[[Conversation], Awaitable[dict[str, Any]]],
    concurrency: int,
    cache: bool,
    rpm: float | None = None,
) -> pd.DataFrame:
    """Shared driver: bounded concurrency, optional requests-per-minute pacing, per-conversation
    disk cache, errors kept as rows."""
    # ponytail: one JSON file per result; move to a parquet/duckdb store if caches reach ~1e5 rows
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    next_start = 0.0

    async def pace() -> None:
        nonlocal next_start
        if not rpm:
            return
        async with lock:
            await asyncio.sleep(max(0.0, next_start - time.monotonic()))
            next_start = max(next_start, time.monotonic()) + 60 / rpm

    async def run(conv: Conversation) -> dict[str, Any]:
        path: Path = CACHE_DIR / backend / f"{cache_key(backend, task, model, conv)}.json"
        if cache and path.exists():
            return dict(json.loads(path.read_text()))
        base = {"id": conv.id, "task": task.name, "backend": backend, "model": model}
        async with sem:
            await pace()
            try:
                row = base | await one(conv) | {"error": None}
            except Exception as e:  # noqa: BLE001 - one bad item must not sink the batch
                logger.error("%s failed on %s: %s", model, conv.id, e)
                return base | {"error": f"{type(e).__name__}: {e}"}
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, default=str))
        return row

    df = pd.DataFrame(await asyncio.gather(*(run(c) for c in convs)))
    for field in task.output_type.model_fields:  # keep schema columns even if every row failed
        if field not in df:
            df[field] = None
    return df


async def run_task(
    task: Task,
    convs: Sequence[Conversation],
    model: str | Model,
    *,
    concurrency: int = 8,
    rpm: float | None = None,
    cache: bool = True,
    model_settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run an LLM task. `model` is a pydantic-ai model string, e.g. 'google:gemini-flash-latest'.

    Set `rpm` below the provider quota (Gemini free tier: 15 requests/min per model).
    """
    agent = Agent(model, output_type=task.output_type, instructions=task.instructions)
    settings: Any = {"temperature": 0.0} if model_settings is None else model_settings
    name = model if isinstance(model, str) else model.model_name

    async def one(conv: Conversation) -> dict[str, Any]:
        async def attempt() -> tuple[Any, float]:  # time only the successful call, not backoff
            t0 = time.perf_counter()
            r = await agent.run(conv.to_text(), model_settings=settings)
            return r, time.perf_counter() - t0

        result, latency = await with_retries(attempt)
        usage = result.usage
        return {
            "resolved_model": result.response.model_name,
            **result.output.model_dump(),
            "latency_s": latency,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": None if usage.cost is None else float(usage.cost),
        }

    return await cached_rows(
        "llm", task, name, convs, one, concurrency, cache and isinstance(model, str), rpm
    )
