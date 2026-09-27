"""Offline checks of the run -> eval pipeline (no network): TestModel, FunctionModel, mocked Jev."""

import asyncio
import json

import httpx2
import pandas as pd
import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from typesafe_sdk import AsyncTypeSafeClient

import genaianalysis.run as run_mod
from genaianalysis.backends.jev import questions_for, run_jev
from genaianalysis.eval import ece, score
from genaianalysis.ingest.tabular import from_frame
from genaianalysis.run import run_task
from genaianalysis.tasks.sentiment import MESSAGE_SENTIMENT, SENTIMENT_ARC

CONVS = from_frame(
    pd.DataFrame({"text": ["love it", "hate it", "  ", "meh"], "sentiment": ["positive", "negative", "x", "neutral"]}),
    "text",
    label_cols=["sentiment"],
)  # fmt: skip


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "CACHE_DIR", tmp_path)


def test_from_frame_skips_blank_text():
    assert [c.labels["sentiment"] for c in CONVS] == ["positive", "negative", "neutral"]


def test_run_task_rows_and_errors(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(run_mod.asyncio, "sleep", no_sleep)  # skip retry backoff
    ok = asyncio.run(run_task(MESSAGE_SENTIMENT, CONVS, TestModel()))
    assert (
        ok["error"].isna().all() and ok["sentiment"].isin(["positive", "negative", "neutral"]).all()
    )

    def boom(messages, info):
        raise RuntimeError("provider down")

    bad = asyncio.run(run_task(MESSAGE_SENTIMENT, CONVS[:1], FunctionModel(boom)))
    assert bad.loc[0, "error"].startswith("RuntimeError") and len(bad) == 1


def test_jev_maps_literal_fields_and_parses_answers():
    assert set(questions_for(SENTIMENT_ARC)) == {"initial_sentiment", "final_sentiment"}

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["state"]["conversation"][0]["from"] == "user"
        proba = {"positive": 0.7, "negative": 0.2, "neutral": 0.1}
        return httpx2.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {"sentiment": {"type": "choice", "choice": "positive", "confidence": 0.6, "probabilities": proba}},
            "usage": {"input_tokens": 1000, "output_tokens": 5},
        })  # fmt: skip

    async def go():
        async with AsyncTypeSafeClient(
            api_key="test", transport=httpx2.MockTransport(handler)
        ) as c:
            return await run_jev(MESSAGE_SENTIMENT, CONVS, client=c)

    df = asyncio.run(go())
    assert (df["sentiment"] == "positive").all() and df["resolved_model"].eq("jev-1.13.0").all()
    assert df["cost_usd"].iloc[0] == pytest.approx(1000 * 0.042 / 1e6)

    s = score(df, CONVS, "sentiment").iloc[0]
    assert s["accuracy"] == pytest.approx(1 / 3) and s["coverage"] == 1 and "ece" in s


def test_ece():
    conf = pd.Series([0.9] * 10)
    assert ece(conf, pd.Series([True] * 9 + [False])) == pytest.approx(0.0)
    assert ece(conf, pd.Series([False] * 10)) == pytest.approx(0.9)


def test_jev_choice_uses_option_rubric():
    from typing import Literal

    from pydantic import BaseModel, Field

    from genaianalysis.schema import Task

    class Out(BaseModel):
        intent: Literal["a", "b"] = Field(
            description="Intent", json_schema_extra={"criteria": {"a": "about A"}}
        )

    q = questions_for(Task("t", Out, "x"))["intent"]
    assert q.criteria == {"a": "about A", "b": None}
