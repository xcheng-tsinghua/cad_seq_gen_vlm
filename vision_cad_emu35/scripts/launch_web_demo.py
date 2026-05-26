from __future__ import annotations

import argparse
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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_bootstrap_thread_env()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.utils.runtime_env import normalize_thread_env

normalize_thread_env()

from vision_cad_emu35.config import load_config
from vision_cad_emu35.model_paths import apply_model_root_override
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the remote-accessible RAG web demo.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--kb-dir", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    apply_model_root_override(config.model, args.model_root)
    if args.kb_dir:
        config.rag.kb_dir = args.kb_dir
    from vision_cad_emu35.api.server import create_app

    app = create_app(config)
    import uvicorn

    print(f"Using KB directory: {config.rag.kb_dir}")
    print(f"Open http://SERVER_IP:{config.api.port} or http://{config.api.host}:{config.api.port}")
    uvicorn.run(app, host=config.api.host, port=config.api.port)


if __name__ == "__main__":
    main()
