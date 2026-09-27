"""Hugging Face datasets via the Hub's auto-converted parquet (no `datasets` dependency)."""

from collections.abc import Sequence

import pandas as pd

from genaianalysis.ingest.tabular import from_frame
from genaianalysis.schema import Conversation

PARQUET_URL = "https://huggingface.co/datasets/{dataset}/resolve/refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet"


def load_hf(
    dataset: str,
    config: str,
    split: str,
    *,
    text_col: str = "text",
    label_col: str = "label",
    label_names: Sequence[str] | None = None,
    label_field: str = "sentiment",
    n: int | None = None,
    seed: int = 0,
) -> list[Conversation]:
    """Load a text-classification split; gold label stored under `label_field` (the task field).

    `label_names` maps integer ClassLabels to strings (index -> name).
    """
    # ponytail: first parquet shard only (fine for eval-sized splits); iterate shards for big ones
    df = pd.read_parquet(PARQUET_URL.format(dataset=dataset, config=config, split=split))
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    labels = df[label_col]
    df = df.assign(
        **{
            label_field: labels.map(lambda i: label_names[i]) if label_names else labels.astype(str)
        },
        _id=[f"{config}-{split}-{i}" for i in df.index],
    )
    convs = from_frame(df, text_col, id_col="_id", label_cols=[label_field], source=dataset)
    for c in convs:
        c.meta["config"] = config
    return convs


def tweet_sentiment(
    language: str = "spanish", split: str = "test", n: int | None = 200, seed: int = 0
) -> list[Conversation]:
    """cardiffnlp/tweet_sentiment_multilingual: 3-class tweet sentiment in 8 languages."""
    return load_hf(
        "cardiffnlp/tweet_sentiment_multilingual", language, split,
        label_names=["negative", "neutral", "positive"], n=n, seed=seed,
    )  # fmt: skip
