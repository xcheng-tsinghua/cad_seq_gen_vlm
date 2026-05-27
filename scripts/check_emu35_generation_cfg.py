from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any


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

from config import AppConfig, load_config, resolve_project_path
from models.emu35_adapter import (
    EMU35_REQUIRED_CFG_FIELDS,
    build_emu35_generation_cfg,
    inspect_generation_utils_cfg_fields,
)
from utils.runtime_env import normalize_thread_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate the Emu3.5 generation config object.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    args = parser.parse_args()

    normalize_thread_env()
    config = load_config_with_minimal_fallback(Path(args.config))
    cfg = build_emu35_generation_cfg(config.generation)

    generation_utils_path = None
    resolved_emu_repo_path = resolve_project_path(config.model.emu_repo_path)
    if resolved_emu_repo_path:
        candidate = resolved_emu_repo_path / "src" / "utils" / "generation_utils.py"
        if candidate.exists():
            generation_utils_path = candidate

    inspected: dict[str, list[str]] | None = None
    if generation_utils_path:
        inspected = inspect_generation_utils_cfg_fields(generation_utils_path)

    print(f"config: {Path(args.config).resolve()}")
    print(f"emu_repo_path: {config.model.emu_repo_path}")
    print(f"resolved_emu_repo_path: {resolved_emu_repo_path}")
    print(f"generation_utils.py: {generation_utils_path if generation_utils_path else '<not found>'}")
    if inspected is not None:
        print("generation_utils_cfg_reads:")
        print(json.dumps(inspected, indent=2))

    cfg_dict = vars(cfg)
    print("built_emu35_generation_cfg:")
    print(json.dumps(cfg_dict, indent=2, default=str))

    missing = [field for field in EMU35_REQUIRED_CFG_FIELDS if not hasattr(cfg, field)]
    sampling_missing: list[str] = []
    if inspected is not None:
        sampling_missing = [key for key in inspected["sampling_param_keys"] if key not in cfg.sampling_params]
    if missing or sampling_missing:
        raise RuntimeError(
            "Emu3.5 generation cfg is incomplete. "
            f"missing_fields={missing}; missing_sampling_params={sampling_missing}"
        )
    print("OK: required Emu3.5 generation config fields are present.")
    return 0


def load_config_with_minimal_fallback(path: Path) -> AppConfig:
    try:
        return load_config(path)
    except ImportError as exc:
        if "PyYAML" not in str(exc):
            raise
        print("WARNING: PyYAML is not installed; using a minimal YAML reader for model/generation sections.")
        config = AppConfig()
        section: str | None = None
        section_fields = {
            "model": set(field.name for field in fields(config.model)),
            "generation": set(field.name for field in fields(config.generation)),
        }
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if re.match(r"^\S", line):
                key = line.strip().rstrip(":")
                section = key if key in section_fields else None
                continue
            if section is None:
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            if key in section_fields[section]:
                setattr(getattr(config, section), key, coerce_scalar(raw_value.strip()))
        return config


def coerce_scalar(value: str) -> object:
    if value in {"", "null", "None", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    quoted = value.strip("\"'")
    try:
        return int(quoted)
    except ValueError:
        pass
    try:
        return float(quoted)
    except ValueError:
        return quoted


if __name__ == "__main__":
    raise SystemExit(main())
