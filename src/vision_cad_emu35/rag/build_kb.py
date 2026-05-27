from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from vision_cad_emu35.config import RagConfig
from vision_cad_emu35.data.manifest import load_manifest
from vision_cad_emu35.data.scan_dataset import scan_dataset
from vision_cad_emu35.rag.image_embedding import create_image_embedder
from vision_cad_emu35.rag.kb_schema import KBItem
from vision_cad_emu35.rag.vector_store import build_vector_store
from vision_cad_emu35.utils.jsonl import write_jsonl


def build_kb_from_dataset(
    dataset_root: str | Path | None,
    kb_dir: str | Path,
    config: RagConfig,
    manifest_path: str | Path | None = None,
    validate_images: bool = True,
) -> dict[str, Any]:
    """Build a RAG KB from a CAD rollback dataset or manifest."""
    out = Path(kb_dir)
    out.mkdir(parents=True, exist_ok=True)

    if manifest_path:
        samples = load_manifest(manifest_path)
        issues = []
        scan_stats: dict[str, Any] = {}
    elif dataset_root:
        try:
            scan = scan_dataset(dataset_root, add_stop_samples=False, validate_images=validate_images)
            samples = scan.samples
            issues = [issue.__dict__ for issue in scan.issues]
            scan_stats = scan.stats
        except FileNotFoundError:
            samples = []
            issues = [{"path": str(dataset_root), "issue": "Dataset root not found", "severity": "warning"}]
            scan_stats = {}
    else:
        samples = []
        issues = []
        scan_stats = {}

    samples = [sample for sample in samples if not sample.get("is_stop_sample", False)]
    items = [KBItem.from_sample(sample) for sample in samples]
    embedder = create_image_embedder(config.embedding_backend)
    embeddings: list[np.ndarray] = []
    failed: list[dict[str, str]] = []

    kept_items: list[KBItem] = []
    for item in items:
        try:
            embeddings.append(embedder.embed_pair(item.final_snapshot_path, item.prev_depth_map_path))
            item.metadata["image_embedding_index"] = len(embeddings) - 1
            item.metadata["image_embedding_file"] = "embeddings.npy"
            kept_items.append(item)
        except Exception as exc:
            failed.append({"sample_id": item.sample_id, "error": str(exc)})

    store = build_vector_store(
        embeddings,
        metadata={
            "embedding_backend": config.embedding_backend,
            "vector_backend": config.vector_backend,
        },
    )
    store.save(out)
    faiss_written = maybe_write_faiss_index(out, store.embeddings) if config.vector_backend == "faiss" else False
    write_jsonl(out / "kb_items.jsonl", [item.to_dict() for item in kept_items])

    hist = Counter(item.operation_type for item in kept_items)
    report = {
        "kb_dir": str(out),
        "num_items": len(kept_items),
        "num_failed_items": len(failed),
        "embedding_shape": list(store.shape),
        "faiss_index_written": faiss_written,
        "operation_type_histogram": dict(sorted(hist.items())),
        "scan_stats": scan_stats,
        "issues": issues,
        "failed_items": failed,
        "empty": len(kept_items) == 0,
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def create_empty_kb(kb_dir: str | Path, config: RagConfig | None = None) -> dict[str, Any]:
    cfg = config or RagConfig()
    out = Path(kb_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "kb_items.jsonl", [])
    build_vector_store([], metadata={"embedding_backend": cfg.embedding_backend, "vector_backend": cfg.vector_backend}).save(out)
    report = {
        "kb_dir": str(out),
        "num_items": 0,
        "num_failed_items": 0,
        "embedding_shape": [0, 0],
        "operation_type_histogram": {},
        "issues": [],
        "failed_items": [],
        "empty": True,
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def maybe_write_faiss_index(kb_dir: Path, embeddings: np.ndarray) -> bool:
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        return False
    try:
        import faiss
    except Exception:
        return False
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(kb_dir / "faiss.index"))
    return True
