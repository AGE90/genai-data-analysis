import asyncio

import numpy as np
import pandas as pd
from pydantic_ai.embeddings import TestEmbeddingModel
from pydantic_ai.models.test import TestModel
from sklearn.preprocessing import normalize

import genaianalysis.run as run_mod
from genaianalysis.ingest.tabular import from_frame
from genaianalysis.topics import cluster, embed, label_clusters


def test_cluster_picks_k_by_silhouette():
    rng = np.random.default_rng(0)
    centers = np.eye(3) * 10
    X = normalize(np.vstack([c + rng.normal(size=(20, 3)) for c in centers]))
    labels, scores = cluster(X, k_range=range(2, 7))
    assert len(set(labels)) == 3 and scores["silhouette"].idxmax() == 1  # k=3
    assert all(len(set(labels[i * 20 : (i + 1) * 20])) == 1 for i in range(3))


def test_embed_and_label_clusters(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "CACHE_DIR", tmp_path)
    convs = from_frame(pd.DataFrame({"text": ["a", "b", "c", "d"]}), "text")
    X = asyncio.run(embed([c.to_text() for c in convs], TestEmbeddingModel()))
    assert X.shape[0] == 4 and np.allclose(np.linalg.norm(X, axis=1), 1)
    out = asyncio.run(label_clusters(convs, np.array([0, 0, 1, 1]), TestModel(), X=X))
    assert out["size"].tolist() == [2, 2] and out["error"].isna().all()
