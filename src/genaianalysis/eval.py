"""Score run results against gold labels so every model/technique is compared on the same data."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from genaianalysis.schema import Conversation


def with_gold(results: pd.DataFrame, convs: Sequence[Conversation], field: str) -> pd.DataFrame:
    gold = pd.DataFrame({"id": [c.id for c in convs], "gold": [c.labels.get(field) for c in convs]})
    return results.merge(gold, on="id", how="inner")


def ece(confidence: pd.Series, correct: pd.Series, bins: int = 10) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over equal-width bins."""
    idx = np.minimum((confidence.to_numpy() * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": idx, "conf": confidence.to_numpy(), "ok": correct.to_numpy()})
    g = df.groupby("bin").agg(conf=("conf", "mean"), acc=("ok", "mean"), n=("ok", "size"))
    return float((g["n"] * (g["acc"] - g["conf"]).abs()).sum() / len(df))


def score(results: pd.DataFrame, convs: Sequence[Conversation], field: str) -> pd.DataFrame:
    """One row per model: quality (on answered items), coverage, latency and cost.

    If the backend returns `<field>_proba` dicts (Jev), also reports ECE of the top probability.
    """
    df = with_gold(results, convs, field).dropna(subset=["gold"])
    rows = []
    for (backend, model), g in df.groupby(["backend", "model"], dropna=False):
        ok = g[g["error"].isna()] if "error" in g else g
        y, p = ok["gold"].astype(str), ok[field].astype(str)
        row = {
            "backend": backend,
            "model": model,
            "resolved_model": ", ".join(sorted(ok["resolved_model"].dropna().unique()))
            if "resolved_model" in ok
            else None,
            "n": len(g),
            "coverage": len(ok) / len(g),
            "accuracy": accuracy_score(y, p) if len(ok) else np.nan,
            "macro_f1": f1_score(y, p, average="macro") if len(ok) else np.nan,
            "kappa": cohen_kappa_score(y, p) if len(ok) else np.nan,
            "latency_p50_s": ok["latency_s"].median(),
            "cost_usd_per_1k": ok["cost_usd"].mean() * 1000
            if ok["cost_usd"].notna().any()
            else np.nan,
        }
        if f"{field}_proba" in ok and ok[f"{field}_proba"].notna().any():
            top = ok[f"{field}_proba"].map(lambda d: max(d.values()))
            row["ece"] = ece(top, y == p)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False, ignore_index=True)
