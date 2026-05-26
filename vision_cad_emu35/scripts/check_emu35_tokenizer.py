from __future__ import annotations

import argparse
import json
import os
import re
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

from vision_cad_emu35.config import AppConfig, load_config, resolve_project_path
from vision_cad_emu35.model_paths import apply_model_root_override, ensure_default_local_model_paths
from vision_cad_emu35.models.emu35_compat import apply_emu3_tokenizer_compat, is_special_tokens_set_error
from vision_cad_emu35.utils.runtime_env import normalize_thread_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Emu3.5 tokenizer compatibility with this Transformers runtime.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--no-clear-cache", action="store_true")
    args = parser.parse_args()

    normalize_thread_env(verbose=True)
    config = load_config_with_minimal_fallback(Path(args.config))
    apply_model_root_override(config.model, args.model_root)
    ensure_default_local_model_paths(config.model)

    print(f"sys.executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print_package_version("torch")
    print_package_version("transformers")
    print(f"tokenizer_path: {config.model.tokenizer_path}")
    print(f"model_id_or_path: {config.model.model_id_or_path}")
    print(f"emu_repo_path: {config.model.emu_repo_path}")
    print(f"resolved_emu_repo_path: {resolve_project_path(config.model.emu_repo_path)}")

    report = apply_emu3_tokenizer_compat(
        config.model,
        clear_cache=False if args.no_clear_cache else config.model.clear_transformers_remote_code_cache,
        verbose=True,
    )
    print("tokenizer_compat_report:")
    print(json.dumps(report.to_dict(), indent=2, default=str))

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_path or config.model.model_id_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        if is_special_tokens_set_error(exc):
            raise RuntimeError(
                "Emu3Tokenizer failed during initialization because special_tokens_set is missing. "
                "The compatibility patch was not applied or stale Transformers remote-code cache is still being used."
            ) from exc
        raise RuntimeError(f"Failed to load Emu3.5 tokenizer: {type(exc).__name__}: {exc}") from exc

    print(f"tokenizer class: {tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}")
    try:
        print(f"vocab size: {len(tokenizer)}")
    except Exception as exc:
        print(f"vocab size: unavailable ({exc})")
    print(f"special_tokens_map: {getattr(tokenizer, 'special_tokens_map', {})}")

    if tokenizer.__class__.__name__ == "Emu3Tokenizer":
        has_safe_path = hasattr(tokenizer, "_get_emu3_special_tokens_set") or hasattr(tokenizer, "special_tokens_set")
        if not has_safe_path:
            raise RuntimeError(
                "Loaded Emu3Tokenizer but it still has no safe special-token path. "
                "Ensure tokenization_emu3.py was patched and stale remote-code cache was cleared."
            )

    text = "CAD preview tokenizer check."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"encoded length: {len(encoded)}")
    print(f"decoded: {decoded}")
    print("OK: Emu3.5 tokenizer loaded and basic encode/decode works.")
    return 0


def print_package_version(name: str) -> None:
    try:
        module = __import__(name)
    except Exception as exc:
        print(f"{name} import status: failed ({exc})")
        return
    print(f"{name} version: {getattr(module, '__version__', 'unknown')}")


def load_config_with_minimal_fallback(path: Path) -> AppConfig:
    try:
        return load_config(path)
    except ImportError as exc:
        if "PyYAML" not in str(exc):
            raise
        print("WARNING: PyYAML is not installed; using a minimal YAML reader for the model section only.")
        config = AppConfig()
        in_model = False
        known = set(config.model.__dataclass_fields__)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if re.match(r"^\S", line):
                in_model = line.strip() == "model:"
                continue
            if not in_model:
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            if key in known:
                setattr(config.model, key, coerce_scalar(raw_value.strip()))
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
        return quoted


if __name__ == "__main__":
    raise SystemExit(main())
