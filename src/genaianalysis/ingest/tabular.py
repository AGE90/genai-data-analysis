"""Generic tabular sources (CSV / JSONL / parquet, surveys, tickets, reviews) -> Conversations."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from genaianalysis.schema import Conversation, Message


def read_table(path: Path | str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {suffix}")


def from_frame(
    df: pd.DataFrame,
    text_col: str,
    *,
    id_col: str | None = None,
    label_cols: Sequence[str] = (),
    sender: str = "user",
    source: str = "table",
) -> list[Conversation]:
    """One row -> one single-message Conversation. `label_cols` become gold labels."""
    # ponytail: no multi-row threads yet; add a group_col (+ order_col) when ticket threads show up
    return [
        Conversation(
            id=str(row[id_col]) if id_col else f"{source}-{idx}",
            messages=[Message(sender=sender, text=str(row[text_col]))],
            labels={c: str(row[c]) for c in label_cols},
            meta={"source": source},
        )
        for idx, row in df.iterrows()
        if isinstance(row[text_col], str) and row[text_col].strip()
    ]
