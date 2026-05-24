from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.config import load_config
from vision_cad_emu35.inference.single_step import load_adapter_for_inference, run_single_step
from vision_cad_emu35.utils.jsonl import read_jsonl, write_jsonl
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference from a manifest JSONL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    adapter = load_adapter_for_inference(config, args.checkpoint)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, row in enumerate(read_jsonl(args.manifest)):
        sample_dir = out / f"{idx:05d}_{row.get('sample_id', 'sample')}"
        result = run_single_step(
            adapter,
            row["final_snapshot_path"],
            row["prev_depth_map_path"],
            sample_dir,
            row.get("prompt"),
            config.generation,
        )
        results.append({"sample_id": row.get("sample_id"), "operation_type": result["operation_type"], "output_dir": str(sample_dir)})
    write_jsonl(out / "batch_results.jsonl", results)
    print(json.dumps({"num_results": len(results), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

