from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.config import load_config
from vision_cad_emu35.inference.single_step import load_adapter_for_inference, run_single_step
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CAD planner step.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--final-snapshot", required=True)
    parser.add_argument("--prev-depth-map", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    adapter = load_adapter_for_inference(config, args.checkpoint)
    result = run_single_step(
        adapter,
        args.final_snapshot,
        args.prev_depth_map,
        args.output_dir,
        args.prompt,
        config.generation,
    )
    print({"operation_type": result["operation_type"], "output_dir": result["output_dir"]})


if __name__ == "__main__":
    main()

