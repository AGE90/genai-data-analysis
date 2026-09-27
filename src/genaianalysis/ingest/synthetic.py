"""Synthetic customer-service chats with known sentiment arcs (ground truth by construction).

Caveat: gold labels are what the generator was *asked* for; a generator that ignores the brief
produces label noise, and a judge from the same model family may be favoured. Spot-check samples.
"""

import asyncio
import itertools
import logging
import random
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from genaianalysis.run import with_retries
from genaianalysis.schema import Conversation, Message

logger = logging.getLogger(__name__)

DOMAINS = [
    "mobile phone and data plans",
    "online banking and card payments",
    "e-commerce order delivery",
    "internet and TV service outages",
    "online betting and lottery platform recharges",
    "utility bills (electricity, water)",
    "health insurance appointments",
    "airline booking changes",
]
ARCS = list(itertools.product(["positive", "negative", "neutral"], repeat=2))


class _Turn(BaseModel):
    sender: Literal["customer", "agent"]
    text: str


class _Chat(BaseModel):
    messages: list[_Turn]


PROMPT = """Write a realistic WhatsApp customer-service chat.
Domain: {domain}. Language: {language}.
The customer's sentiment is {initial} at the beginning and {final} at the end.
Between {min_turns} and {max_turns} messages, customer first. Use informal chat style (short
messages, occasional typos, emojis where natural). Invent plausible but fake names and IDs.
Never state the sentiment labels explicitly; show them through what people say."""


async def generate_chats(
    n: int,
    model: str | Model,
    *,
    language: str = "Spanish",
    seed: int = 0,
    concurrency: int = 4,
    rpm: float | None = None,
    min_turns: int = 4,
    max_turns: int = 12,
) -> list[Conversation]:
    """Generate `n` chats with sentiment arcs balanced over the 9 (initial, final) combinations."""
    rng = random.Random(seed)
    domains = [rng.choice(DOMAINS) for _ in range(n)]
    agent = Agent(model, output_type=_Chat)
    sem = asyncio.Semaphore(concurrency)
    delay = 60 / rpm if rpm else 0.0
    name = model if isinstance(model, str) else model.model_name

    async def one(i: int) -> Conversation | None:
        initial, final = ARCS[i % len(ARCS)]
        spec: dict[str, Any] = dict(
            domain=domains[i], language=language, initial=initial, final=final,
            min_turns=min_turns, max_turns=max_turns,
        )  # fmt: skip
        await asyncio.sleep(i * delay)  # staggered starts keep us under `rpm`
        async with sem:
            try:
                r = await with_retries(
                    lambda: agent.run(PROMPT.format(**spec), model_settings={"temperature": 1.0})
                )
            except Exception as e:  # noqa: BLE001 - skip failed generations, keep the rest
                logger.error("generation %d failed: %s", i, e)
                return None
        return Conversation(
            id=f"syn-{seed}-{i}",
            messages=[Message(sender=t.sender, text=t.text) for t in r.output.messages],
            labels={"initial_sentiment": initial, "final_sentiment": final},
            meta={"source": "synthetic", "generator": name, "domain": spec["domain"],
                  "language": language},
        )  # fmt: skip

    return [c for c in await asyncio.gather(*(one(i) for i in range(n))) if c]
