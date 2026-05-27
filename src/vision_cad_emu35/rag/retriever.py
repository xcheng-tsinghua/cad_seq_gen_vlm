from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from vision_cad_emu35.config import RagConfig
from vision_cad_emu35.rag.image_embedding import create_image_embedder
from vision_cad_emu35.rag.kb_schema import KBItem, item_from_dict
from vision_cad_emu35.rag.vector_store import NumpyVectorStore
from vision_cad_emu35.utils.jsonl import read_jsonl


class RagRetriever:
    def __init__(self, kb_dir: str | Path, config: RagConfig | dict[str, Any] | None = None) -> None:
        self.kb_dir = Path(kb_dir).expanduser()
        self.config = config if isinstance(config, RagConfig) else RagConfig(**(config or {}))
        self.embedder = create_image_embedder(self.config.embedding_backend)
        self.items: list[KBItem] = []
        self.store = NumpyVectorStore()
        self.load()

    def load(self) -> None:
        items_path = self.kb_dir / "kb_items.jsonl"
        if items_path.exists():
            self.items = [item_from_dict(row) for row in read_jsonl(items_path)]
        else:
            self.items = []
        try:
            self.store = NumpyVectorStore.load(self.kb_dir)
        except Exception:
            self.store = NumpyVectorStore()
        if len(self.items) != self.store.shape[0]:
            self.items = self.items[: self.store.shape[0]]

    def is_empty(self) -> bool:
        return len(self.items) == 0 or self.store.is_empty

    def item_count(self) -> int:
        return 0 if self.is_empty() else len(self.items)

    def status(self) -> dict[str, Any]:
        return {
            "kb_dir": str(self.kb_dir),
            "kb_loaded": True,
            "kb_empty": self.is_empty(),
            "kb_item_count": self.item_count(),
            "embedding_shape": list(self.store.shape),
        }

    def retrieve(
        self,
        final_snapshot: Image.Image,
        prev_depth_map: Image.Image,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.is_empty():
            return []
        filters = filters or {}
        operation_type = filters.get("operation_type") or filters.get("operation_type_hint")
        candidate_indices = None
        if operation_type:
            candidate_indices = [idx for idx, item in enumerate(self.items) if item.operation_type == operation_type]
            if not candidate_indices:
                return []

        query = self.embedder.embed_pair(final_snapshot, prev_depth_map)
        hits = self.store.search(query, top_k=top_k, candidate_indices=candidate_indices)
        results: list[dict[str, Any]] = []
        for idx, score in hits:
            item = self.items[idx]
            row = item.to_dict()
            row["score"] = score
            row["metadata"] = dict(row.get("metadata") or {})
            row["metadata"]["kb_dir"] = str(self.kb_dir)
            results.append(row)
        return results
