from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.config import load_config
from vision_cad_emu35.model_paths import apply_model_root_override, ensure_default_local_model_paths, validate_local_model_paths
from vision_cad_emu35.models.emu35_adapter import Emu35Adapter
from vision_cad_emu35.train.trainer import Emu35Trainer
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Emu3.5 as a CAD planner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-root", default=None, help="Override local model root and derive Emu3.5 paths.")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    apply_model_root_override(config.model, args.model_root)
    ensure_default_local_model_paths(config.model)
    validate_local_model_paths(config.model)
    adapter = Emu35Adapter(config.model)
    Emu35Trainer(config, adapter).train()


if __name__ == "__main__":
    main()
