from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ROOT = "/root/autodl-tmp/data"
DEFAULT_MAIN_LOCAL_ID = "BAAI/Emu3.5"
DEFAULT_VISION_TOKENIZER_LOCAL_ID = "BAAI/Emu3.5-VisionTokenizer"
DEFAULT_MAIN_MODELSCOPE_ID = "BAAI/Emu3.5"
DEFAULT_VISION_TOKENIZER_MODELSCOPE_ID = "BAAI/Emu3.5-VisionTokenizer"
DEFAULT_MAIN_HF_REPO_ID = "BAAI/Emu3.5"
DEFAULT_VISION_TOKENIZER_HF_REPO_ID = "BAAI/Emu3.5-VisionTokenizer"
DEFAULT_MAIN_REPO_ID = DEFAULT_MAIN_LOCAL_ID
DEFAULT_VISION_TOKENIZER_REPO_ID = DEFAULT_VISION_TOKENIZER_LOCAL_ID
DOWNLOAD_COMMAND = "python scripts/download_models.py"


def repo_id_to_local_path(model_root: str | Path, repo_id: str) -> Path:
    """Map a stable local model id to the local directory layout under model_root."""
    return Path(local_repo_path_string(model_root, repo_id)).expanduser()


def local_repo_path_string(model_root: str | Path, repo_id: str) -> str:
    root = Path(model_root).expanduser().as_posix()
    return f"{root.rstrip('/')}/{repo_id}"


def default_local_model_paths(
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    main_repo_id: str = DEFAULT_MAIN_REPO_ID,
    vision_tokenizer_repo_id: str = DEFAULT_VISION_TOKENIZER_REPO_ID,
) -> dict[str, str]:
    root = Path(model_root).expanduser().as_posix()
    main_path = local_repo_path_string(model_root, main_repo_id)
    vision_path = local_repo_path_string(model_root, vision_tokenizer_repo_id)
    return {
        "model_root": root,
        "model_id_or_path": main_path,
        "tokenizer_path": main_path,
        "vision_tokenizer_path": vision_path,
    }


def apply_model_root_override(
    model_config: Any,
    model_root: str | Path | None,
    main_repo_id: str = DEFAULT_MAIN_REPO_ID,
    vision_tokenizer_repo_id: str = DEFAULT_VISION_TOKENIZER_REPO_ID,
) -> Any:
    """Apply --model-root by deriving all runtime model paths from that root."""
    if model_root is None:
        return model_config
    paths = default_local_model_paths(model_root, main_repo_id, vision_tokenizer_repo_id)
    _set_attr_or_key(model_config, "model_root", paths["model_root"])
    _set_attr_or_key(model_config, "model_id_or_path", paths["model_id_or_path"])
    _set_attr_or_key(model_config, "tokenizer_path", paths["tokenizer_path"])
    _set_attr_or_key(model_config, "vision_tokenizer_path", paths["vision_tokenizer_path"])
    return model_config


def ensure_default_local_model_paths(model_config: Any) -> Any:
    """Fill missing model paths from model_root and keep runtime defaults local."""
    model_root = _get_attr_or_key(model_config, "model_root") or DEFAULT_MODEL_ROOT
    paths = default_local_model_paths(model_root)
    local_files_only = _get_attr_or_key(model_config, "local_files_only")
    _set_attr_or_key(model_config, "model_root", paths["model_root"])
    for key in ("model_id_or_path", "tokenizer_path", "vision_tokenizer_path"):
        value = _get_attr_or_key(model_config, key)
        if not value or (local_files_only is not False and value in {DEFAULT_MAIN_REPO_ID, DEFAULT_VISION_TOKENIZER_REPO_ID}):
            _set_attr_or_key(model_config, key, paths[key])
    if local_files_only is None:
        _set_attr_or_key(model_config, "local_files_only", True)
    return model_config


def validate_local_model_paths(model_config: Any) -> None:
    """Fail early if local-only runtime paths are missing or incomplete."""
    ensure_default_local_model_paths(model_config)
    local_files_only = bool(_get_attr_or_key(model_config, "local_files_only"))
    if not local_files_only:
        return

    model_path = Path(str(_get_attr_or_key(model_config, "model_id_or_path"))).expanduser()
    tokenizer_path = Path(str(_get_attr_or_key(model_config, "tokenizer_path") or model_path)).expanduser()
    vision_path = Path(str(_get_attr_or_key(model_config, "vision_tokenizer_path"))).expanduser()

    errors: list[str] = []
    for label, path in (
        ("model_id_or_path", model_path),
        ("tokenizer_path", tokenizer_path),
        ("vision_tokenizer_path", vision_path),
    ):
        if not path.exists():
            errors.append(f"{label} does not exist: {path}")
        elif not path.is_dir():
            errors.append(f"{label} is not a directory: {path}")

    if model_path.exists():
        if not (model_path / "config.json").exists():
            errors.append(f"config.json not found in main model directory: {model_path}")
        if not _has_weight_file(model_path):
            errors.append(f"no model weight shard found in main model directory: {model_path}")

    if tokenizer_path.exists() and not _has_any_file(tokenizer_path, ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")):
        errors.append(f"no tokenizer_config.json/tokenizer.json/tokenizer.model found in tokenizer directory: {tokenizer_path}")

    if vision_path.exists():
        if not (vision_path / "config.json").exists():
            errors.append(f"config.json not found in vision tokenizer directory: {vision_path}")
        if not _has_weight_file(vision_path):
            errors.append(f"no weight shard found in vision tokenizer directory: {vision_path}")

    if errors:
        joined = "\n  - ".join(errors)
        raise FileNotFoundError(
            "Local Emu3.5 weights not found. Please run: "
            f"{DOWNLOAD_COMMAND}\n  - {joined}"
        )


def model_config_to_yaml_snippet(model_root: str | Path = DEFAULT_MODEL_ROOT) -> str:
    paths = default_local_model_paths(model_root)
    return "\n".join(
        [
            "model:",
            f'  model_root: "{paths["model_root"]}"',
            f'  model_id_or_path: "{paths["model_id_or_path"]}"',
            f'  tokenizer_path: "{paths["tokenizer_path"]}"',
            f'  vision_tokenizer_path: "{paths["vision_tokenizer_path"]}"',
            "  emu_repo_path: null",
            "  local_files_only: true",
        ]
    )


def _has_weight_file(path: Path) -> bool:
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.index.json")
    return any(next(path.rglob(pattern), None) is not None for pattern in patterns)


def _has_any_file(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).exists() for name in names)


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _set_attr_or_key(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def serializable_model_config(model_config: Any) -> dict[str, Any]:
    if is_dataclass(model_config):
        return asdict(model_config)
    if isinstance(model_config, dict):
        return dict(model_config)
    return dict(vars(model_config))
