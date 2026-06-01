from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from filenames import KB_EMBEDDINGS, KB_FAISS_INDEX, KB_ITEMS
from rag.retriever import RagRetriever
from utils.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a RAG knowledge base.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--kb-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    kb_dir = Path(args.kb_dir or config.rag.kb_dir)
    items_path = kb_dir / KB_ITEMS
    items = list(read_jsonl(items_path)) if items_path.exists() else []
    retriever = RagRetriever(kb_dir)
    hist = Counter(item.get("operation_type", "unknown") for item in items)
    summary = {
        "kb_dir": str(kb_dir),
        "num_items": len(items),
        "operation_type_histogram": dict(sorted(hist.items())),
        "embedding_shape": list(retriever.store.shape),
        "vector_index_exists": (kb_dir / KB_FAISS_INDEX).exists() or (kb_dir / KB_EMBEDDINGS).exists(),
        "is_empty": retriever.is_empty(),
        "example_items": items[:3],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
