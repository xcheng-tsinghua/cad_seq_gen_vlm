from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.config import load_config
from vision_cad_emu35.inference.autoregressive import AutoregressiveCADPlanner
from vision_cad_emu35.inference.executor import CopyDepthExecutor
from vision_cad_emu35.inference.single_step import load_adapter_for_inference
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoregressive CAD planner inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--final-snapshot", required=True)
    parser.add_argument("--initial-depth-map", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--teacher-depth-sequence", nargs="*", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    adapter = load_adapter_for_inference(config, args.checkpoint)
    executor = CopyDepthExecutor(args.teacher_depth_sequence) if args.teacher_depth_sequence else None
    planner = AutoregressiveCADPlanner(adapter, executor=executor, generation_config=config.generation)
    steps = planner.run(
        args.final_snapshot,
        args.initial_depth_map,
        args.output_dir,
        max_steps=args.max_steps,
        prompt=args.prompt,
    )
    print(json.dumps({"num_steps": len(steps), "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()

