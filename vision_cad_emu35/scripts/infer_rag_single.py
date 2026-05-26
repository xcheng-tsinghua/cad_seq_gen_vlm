from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            valid = int(str(os.environ.get(name, "")).strip()) > 0
        except ValueError:
            valid = False
        if not valid:
            os.environ[name] = "8"


_bootstrap_thread_env()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.utils.runtime_env import normalize_thread_env

normalize_thread_env()

from vision_cad_emu35.config import load_config
from vision_cad_emu35.inference.rag_single_step import load_frozen_adapter, run_rag_single_step
from vision_cad_emu35.model_paths import apply_model_root_override
from vision_cad_emu35.rag.retriever import RagRetriever
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen Emu3.5 RAG CAD planning step.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--final-snapshot", required=True)
    parser.add_argument("--prev-depth-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--prompt-extra", default=None)
    parser.add_argument("--operation-type-hint", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--kb-dir", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    apply_model_root_override(config.model, args.model_root)
    if args.kb_dir:
        config.rag.kb_dir = args.kb_dir
    retriever = RagRetriever(config.rag.kb_dir, config.rag)
    adapter = load_frozen_adapter(config)
    result = run_rag_single_step(
        config,
        adapter,
        retriever,
        args.final_snapshot,
        args.prev_depth_map,
        args.output_dir,
        top_k=args.top_k,
        prompt_extra=args.prompt_extra,
        operation_type_hint=args.operation_type_hint,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "image"}, indent=2, default=str))


if __name__ == "__main__":
    main()
