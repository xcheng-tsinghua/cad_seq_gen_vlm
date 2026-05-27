from __future__ import annotations

import numpy as np

from rag.vector_store import NumpyVectorStore, build_vector_store


def test_vector_store_top_k_retrieval():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    store = build_vector_store(embeddings)
    hits = store.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)
    assert hits[0][0] == 0
    assert len(hits) == 2


def test_empty_vector_store_works():
    store = NumpyVectorStore()
    assert store.is_empty
    assert store.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=3) == []


def test_vector_store_save_load_empty(tmp_path):
    store = NumpyVectorStore(np.zeros((0, 8), dtype=np.float32))
    store.save(tmp_path)
    loaded = NumpyVectorStore.load(tmp_path)
    assert loaded.is_empty
    assert loaded.shape == (0, 8)

