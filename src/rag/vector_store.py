from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rag.image_embedding import l2_normalize


class NumpyVectorStore:
    def __init__(self, embeddings: np.ndarray | None = None, metadata: dict[str, Any] | None = None) -> None:
        if embeddings is None:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must have shape [n, dim], got {embeddings.shape}")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.metadata = metadata or {}

    @property
    def is_empty(self) -> bool:
        return self.embeddings.shape[0] == 0

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.embeddings.shape)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 3,
        candidate_indices: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self.is_empty or top_k <= 0:
            return []
        query = l2_normalize(query.astype(np.float32, copy=False))
        if candidate_indices is None:
            matrix = self.embeddings
            base_indices = np.arange(self.embeddings.shape[0])
        else:
            if not candidate_indices:
                return []
            base_indices = np.asarray(candidate_indices, dtype=np.int64)
            matrix = self.embeddings[base_indices]
        if matrix.shape[1] != query.shape[0]:
            raise ValueError(f"Query dim {query.shape[0]} does not match store dim {matrix.shape[1]}")
        sims = matrix @ query
        order = np.argsort(-sims)[:top_k]
        return [(int(base_indices[i]), float(sims[i])) for i in order]

    def save(self, kb_dir: str | Path) -> None:
        out = Path(kb_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings)
        meta = {
            "backend": "numpy",
            "embedding_shape": list(self.embeddings.shape),
            **self.metadata,
        }
        (out / "vector_store_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, kb_dir: str | Path) -> "NumpyVectorStore":
        root = Path(kb_dir)
        embeddings_path = root / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(0, int(embeddings.shape[0])) if embeddings.size == 0 else embeddings.reshape(1, -1)
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        meta_path = root / "vector_store_meta.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(embeddings=embeddings.astype(np.float32), metadata=metadata)


def build_vector_store(embeddings: list[np.ndarray] | np.ndarray, metadata: dict[str, Any] | None = None) -> NumpyVectorStore:
    if isinstance(embeddings, list):
        if embeddings:
            arr = np.stack([l2_normalize(e) for e in embeddings]).astype(np.float32)
        else:
            arr = np.zeros((0, 0), dtype=np.float32)
    else:
        arr = embeddings.astype(np.float32)
        if arr.size:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
    return NumpyVectorStore(arr, metadata=metadata)
