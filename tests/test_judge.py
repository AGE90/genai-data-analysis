import asyncio

import pandas as pd
from pydantic_ai.models.test import TestModel
from typesafe_sdk import Choice, Noul

import genaianalysis.run as run_mod
from genaianalysis.backends.jev import questions_for
from genaianalysis.ingest.tabular import from_frame
from genaianalysis.run import run_task
from genaianalysis.tasks.judge import (
    ARC_JUDGE,
    CANDIDATE,
    FINAL_SENTIMENT_PAIRWISE,
    with_candidate,
    with_options,
)

(CONV,) = from_frame(pd.DataFrame({"text": ["mi pedido no llega"]}), "text")


def test_with_candidate_appends_output_and_keeps_original(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "CACHE_DIR", tmp_path)
    judged = with_candidate(CONV, {"final_sentiment": "negative"}, "orig")
    assert judged.id == f"{CONV.id}:orig" and len(CONV.messages) == 1
    assert f"\n{CANDIDATE}: {{" in judged.to_text()
    df = asyncio.run(run_task(ARC_JUDGE, [judged], TestModel()))
    assert df["error"].isna().all() and df["score"].isin([1, 2, 3, 4, 5]).all()


def test_with_options_and_jev_mapping():
    text = with_options(CONV, "Final sentiment?", "neutral", "negative", "ab").to_text()
    assert "A: neutral\nB: negative" in text
    q = questions_for(ARC_JUDGE)
    assert {k for k, v in q.items() if isinstance(v, Noul)} == {
        "initial_label_supported", "final_label_supported", "justifications_faithful",
    }  # fmt: skip
    assert isinstance(q["score"], Choice) and isinstance(
        questions_for(FINAL_SENTIMENT_PAIRWISE)["choice"], Choice
    )
