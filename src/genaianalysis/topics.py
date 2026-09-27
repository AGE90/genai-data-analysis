"""Unsupervised topic discovery: embed conversations, cluster, let an LLM name each cluster."""

import dataclasses
import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic_ai.embeddings import Embedder, EmbeddingModel
from pydantic_ai.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from genaianalysis import run
from genaianalysis.schema import Conversation, Message
from genaianalysis.tasks.topics import TOPIC_LABEL


async def embed(
    texts: Sequence[str],
    model: str | EmbeddingModel = "google:gemini-embedding-001",
    batch: int = 100,
) -> np.ndarray:
    """L2-normalized embeddings (rows), cached on disk per (model, texts)."""
    name = model if isinstance(model, str) else model.model_name
    key = hashlib.sha256(json.dumps([name, list(texts)]).encode()).hexdigest()[:24]
    path = run.CACHE_DIR / "embeddings" / f"{key}.npy"
    if isinstance(model, str) and path.exists():
        cached: np.ndarray = np.load(path)
        return cached
    embedder = Embedder(model)
    chunks = []
    for i in range(0, len(texts), batch):
        part = list(texts[i : i + batch])

        async def call(part: list[str] = part) -> list[list[float]]:
            return [list(e) for e in (await embedder.embed_documents(part)).embeddings]

        chunks.append(np.asarray(await run.with_retries(call), dtype=np.float32))
    X: np.ndarray = normalize(np.vstack(chunks))
    if isinstance(model, str):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, X)
    return X


def cluster(
    X: np.ndarray, k: int | None = None, k_range: range = range(2, 16), seed: int = 0
) -> tuple[np.ndarray, pd.DataFrame]:
    """KMeans on normalized embeddings (≈ cosine). With k=None pick k by silhouette.

    Returns labels and the silhouette per candidate k (to eyeball the choice)."""
    # ponytail: KMeans forces every item into a cluster; try sklearn.cluster.HDBSCAN when
    # outliers/noise matter or cluster sizes are very uneven
    ks = [k] if k else [c for c in k_range if c < len(X)]
    fits = {c: KMeans(n_clusters=c, n_init=10, random_state=seed).fit(X) for c in ks}
    scores = pd.DataFrame(
        {
            "k": ks,
            "silhouette": [
                silhouette_score(X, fits[c].labels_) if len(ks) > 1 else np.nan for c in ks
            ],
        }
    )
    best = ks[0] if len(ks) == 1 else int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return fits[best].labels_, scores


async def label_clusters(
    convs: Sequence[Conversation],
    labels: np.ndarray,
    model: str | Model,
    *,
    X: np.ndarray | None = None,
    n_examples: int = 8,
    max_chars: int = 600,
    language: str | None = None,
    rpm: float | None = None,
) -> pd.DataFrame:
    """One row per cluster: size, LLM name/description. Examples are the items closest to the
    cluster centroid when embeddings `X` are given, else the first ones. `language` pins the
    language of the names (models otherwise drift between languages across clusters)."""
    samples = []
    for c in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == c)
        if X is not None:
            centroid = X[idx].mean(axis=0)
            idx = idx[np.argsort(-(X[idx] @ centroid))]
        msgs = [
            Message(sender=f"example {i + 1}", text=convs[j].to_text()[:max_chars])
            for i, j in enumerate(idx[:n_examples])
        ]
        samples.append(Conversation(id=f"topic-{c}", messages=msgs))
    task = TOPIC_LABEL
    if language:
        task = dataclasses.replace(task, instructions=f"{task.instructions} Answer in {language}.")
    named = await run.run_task(task, samples, model, rpm=rpm)
    sizes = pd.Series(labels).value_counts()
    named["cluster"] = named["id"].str.removeprefix("topic-").astype(int)
    named["size"] = named["cluster"].map(sizes)
    return named[["cluster", "size", "name", "description", "error"]].sort_values(
        "size", ascending=False, ignore_index=True
    )
