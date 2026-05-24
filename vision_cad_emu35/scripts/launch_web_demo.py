from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.api.server import create_app
from vision_cad_emu35.config import load_config
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local web demo.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "api.yaml"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    app = create_app(config, args.checkpoint)
    import uvicorn

    print(f"Open http://{config.api.host}:{config.api.port}")
    uvicorn.run(app, host=config.api.host, port=config.api.port)


if __name__ == "__main__":
    main()

