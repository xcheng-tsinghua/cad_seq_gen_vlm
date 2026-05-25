from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.config import load_config
from vision_cad_emu35.inference.rag_single_step import load_frozen_adapter, run_rag_single_step
from vision_cad_emu35.model_paths import apply_model_root_override
from vision_cad_emu35.rag.retriever import RagRetriever
from vision_cad_emu35.utils.jsonl import read_jsonl, write_jsonl
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Emu3.5 RAG inference for a manifest JSONL.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--model-root", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    apply_model_root_override(config.model, args.model_root)
    retriever = RagRetriever(config.rag.kb_dir, config.rag)
    adapter = load_frozen_adapter(config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, sample in enumerate(read_jsonl(args.manifest)):
        sample_dir = out / f"{idx:05d}_{sample.get('sample_id', 'sample')}"
        result = run_rag_single_step(
            config,
            adapter,
            retriever,
            sample["final_snapshot_path"],
            sample["prev_depth_map_path"],
            sample_dir,
            top_k=args.top_k,
        )
        rows.append({k: v for k, v in result.items() if k != "image"})
    write_jsonl(out / "batch_results.jsonl", rows)
    print(json.dumps({"num_results": len(rows), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

