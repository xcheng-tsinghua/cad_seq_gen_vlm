from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.config import load_config
from vision_cad_emu35.rag.build_kb import build_kb_from_dataset
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RAG knowledge base from CAD rollback examples.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--no-validate-images", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    dataset_root = args.dataset_root or config.data.dataset_root
    kb_dir = args.kb_dir or config.rag.kb_dir
    report = build_kb_from_dataset(
        dataset_root=dataset_root,
        kb_dir=kb_dir,
        config=config.rag,
        manifest_path=args.manifest,
        validate_images=not args.no_validate_images,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

