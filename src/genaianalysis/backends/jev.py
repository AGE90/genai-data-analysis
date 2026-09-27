"""TypeSafe Jev (System One model) backend: same Task, same row shape as run.run_task.

Jev answers typed questions in a single non-autoregressive pass and returns calibrated
probabilities instead of text. Task fields are mapped to questions:
  Literal[...] -> Choice (label + probabilities + confidence)
  bool         -> Noul   (probability of yes, thresholded at 0.5)
  other types (free-text justifications, ...) are skipped: Jev does not generate text.
Needs TYPESAFE_API_KEY. Docs: https://docs.typesafe.ai
"""

import time
import typing
from collections.abc import Sequence
from typing import Any

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from genaianalysis.run import cached_rows
from genaianalysis.schema import Conversation, Task

USD_PER_INPUT_TOKEN = 0.042 / 1e6  # jev-1.13 list price; output tokens are free


def questions_for(task: Task) -> dict[str, Choice | Noul]:
    questions: dict[str, Choice | Noul] = {}
    for name, field in task.output_type.model_fields.items():
        if typing.get_origin(field.annotation) is typing.Literal:
            options = typing.get_args(field.annotation)
            questions[name] = Choice(
                instructions=field.description, criteria={str(o): None for o in options}
            )
        elif field.annotation is bool:
            questions[name] = Noul(instructions=field.description)
        else:
            continue
        if not field.description:
            raise ValueError(f"{task.name}.{name} needs a Field(description=...) to ask Jev")
    if not questions:
        raise ValueError(f"{task.name} has no Literal/bool fields Jev can answer")
    return questions


def state_for(conv: Conversation) -> dict[str, Any]:
    return {
        "conversation": [
            {"from": m.sender, "text": m.text}
            | ({"time": m.timestamp.isoformat()} if m.timestamp else {})
            for m in conv.messages
        ]
    }


async def run_jev(
    task: Task,
    convs: Sequence[Conversation],
    model: str = "jev-latest",
    *,
    concurrency: int = 16,
    cache: bool = True,
    client: AsyncTypeSafeClient | None = None,
) -> pd.DataFrame:
    """Adds per-field `<field>_confidence` and `<field>_proba` columns next to the prediction."""
    questions = questions_for(task)
    if client is None:
        async with AsyncTypeSafeClient(model=model) as owned:
            return await run_jev(
                task, convs, model, concurrency=concurrency, cache=cache, client=owned
            )

    async def one(conv: Conversation) -> dict[str, Any]:
        t0 = time.perf_counter()  # the SDK retries 429/5xx itself (RetryPolicy)
        resp = await client.system_one(state_for(conv), questions)
        latency = time.perf_counter() - t0
        row: dict[str, Any] = {"resolved_model": resp.model}
        for name, ans in resp.answers.items():
            if ans.type == "choice":
                row |= {
                    name: ans.choice,
                    f"{name}_confidence": ans.confidence,
                    f"{name}_proba": ans.probabilities,
                }
            elif ans.type == "noul":
                row |= {name: ans.noul >= 0.5, f"{name}_proba": ans.noul}
        return row | {
            "latency_s": latency,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cost_usd": (resp.usage.input_tokens or 0) * USD_PER_INPUT_TOKEN,
        }

    return await cached_rows("jev", task, model, convs, one, concurrency, cache)
