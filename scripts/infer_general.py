from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _bootstrap_thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            valid = int(str(os.environ.get(name, "")).strip()) > 0
        except ValueError:
            valid = False
        if not valid:
            os.environ[name] = "8"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_bootstrap_thread_env()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.runtime_env import normalize_thread_env

normalize_thread_env()

from config import load_config
from inference.general import (
    MAX_GENERAL_INPUT_IMAGES,
    clear_cuda_cache,
    cuda_oom_diagnostics,
    is_cuda_oom_error,
    load_general_adapter,
    run_general_inference,
    write_general_error_response,
)
from model_paths import apply_model_root_override
from utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Emu3.5 in general multimodal mode.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "general.yaml"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[], help="Optional input image. Repeat for multiple images.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    apply_model_root_override(config.model, args.model_root)
    if len(args.image) > MAX_GENERAL_INPUT_IMAGES:
        raise SystemExit(f"General Emu3.5 mode accepts at most {MAX_GENERAL_INPUT_IMAGES} input images.")

    start = time.perf_counter()
    try:
        adapter = load_general_adapter(config)
        response = run_general_inference(config, adapter, args.prompt, args.image, args.output_dir)
    except Exception as exc:
        if is_cuda_oom_error(exc):
            clear_cuda_cache()
            diagnostics = cuda_oom_diagnostics(
                exc,
                num_images=len(args.image),
                max_new_tokens=getattr(config.generation, "max_new_tokens", None),
            )
            response = write_general_error_response(
                args.output_dir,
                diagnostics,
                prompt=args.prompt,
                latency_seconds=time.perf_counter() - start,
            )
            print(json.dumps(response, indent=2, default=str))
            raise SystemExit(2) from exc
        raise

    print(json.dumps({k: v for k, v in response.items() if k != "image"}, indent=2, default=str))


if __name__ == "__main__":
    main()
