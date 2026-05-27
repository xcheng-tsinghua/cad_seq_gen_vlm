from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_paths import (
    DEFAULT_MAIN_HF_REPO_ID,
    DEFAULT_MAIN_LOCAL_ID,
    DEFAULT_MAIN_MODELSCOPE_ID,
    DEFAULT_MODEL_ROOT,
    DEFAULT_VISION_TOKENIZER_HF_REPO_ID,
    DEFAULT_VISION_TOKENIZER_LOCAL_ID,
    DEFAULT_VISION_TOKENIZER_MODELSCOPE_ID,
    model_config_to_yaml_snippet,
    repo_id_to_local_path,
)


LOGGER = logging.getLogger("download_models")
MODELSCOPE_OVERRIDE_HINT = (
    "If the default ModelScope ids are unavailable, override them:\n"
    "  python scripts/download_models.py \\\n"
    "    --backend modelscope \\\n"
    "    --main-modelscope-id <actual_modelscope_model_id> \\\n"
    "    --vision-tokenizer-modelscope-id <actual_modelscope_model_id>"
)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    downloaded: list[dict[str, Any]] = []

    main_source_id = args.main_modelscope_id if args.backend == "modelscope" else args.main_hf_repo_id
    vision_source_id = (
        args.vision_tokenizer_modelscope_id
        if args.backend == "modelscope"
        else args.vision_tokenizer_hf_repo_id
    )

    if not args.skip_main_model:
        downloaded.append(
            download_repo(
                backend=args.backend,
                source_id=main_source_id,
                local_layout_id=DEFAULT_MAIN_LOCAL_ID,
                output_dir=output_dir,
                revision=args.revision,
                token=token,
                cache_dir=args.cache_dir,
                resume=args.resume,
                force=args.force,
                local_dir_use_symlinks=parse_bool(args.local_dir_use_symlinks),
                allow_patterns=args.include_pattern,
                ignore_patterns=args.exclude_pattern,
                role="main_model",
                max_retries=args.max_retries,
            )
        )

    if not args.skip_vision_tokenizer:
        downloaded.append(
            download_repo(
                backend=args.backend,
                source_id=vision_source_id,
                local_layout_id=DEFAULT_VISION_TOKENIZER_LOCAL_ID,
                output_dir=output_dir,
                revision=args.revision,
                token=token,
                cache_dir=args.cache_dir,
                resume=args.resume,
                force=args.force,
                local_dir_use_symlinks=parse_bool(args.local_dir_use_symlinks),
                allow_patterns=args.include_pattern,
                ignore_patterns=args.exclude_pattern,
                role="vision_tokenizer",
                max_retries=args.max_retries,
            )
        )

    print()
    print(f"Download backend: {args.backend}")
    print("Downloaded/local model paths:")
    for item in downloaded:
        print(f"- {item['source_id']} -> {item['local_path']}")
    print()
    print("Pasteable runtime config snippet:")
    print(model_config_to_yaml_snippet(output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only downloader for Emu3.5 weights/tokenizers. Defaults to ModelScope, "
            "does not import torch, and does not load the model."
        )
    )
    parser.add_argument("--backend", default="modelscope", choices=["modelscope", "huggingface"])
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Falls back to HF_TOKEN or login cache.")
    parser.add_argument("--main-modelscope-id", default=DEFAULT_MAIN_MODELSCOPE_ID)
    parser.add_argument("--vision-tokenizer-modelscope-id", default=DEFAULT_VISION_TOKENIZER_MODELSCOPE_ID)
    parser.add_argument("--main-hf-repo-id", default=DEFAULT_MAIN_HF_REPO_ID)
    parser.add_argument("--vision-tokenizer-hf-repo-id", default=DEFAULT_VISION_TOKENIZER_HF_REPO_ID)
    parser.add_argument("--main-repo-id", dest="main_hf_repo_id", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--vision-tokenizer-repo-id",
        dest="vision_tokenizer_hf_repo_id",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Delete an existing local repo directory and download again.")
    parser.add_argument("--local-dir-use-symlinks", default="false", choices=["true", "false", "auto"])
    parser.add_argument("--include-pattern", action="append", default=None)
    parser.add_argument("--exclude-pattern", action="append", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--skip-main-model", action="store_true")
    parser.add_argument("--skip-vision-tokenizer", action="store_true")
    return parser


def download_repo(
    backend: str,
    source_id: str,
    local_layout_id: str,
    output_dir: Path,
    revision: str,
    token: str | None,
    cache_dir: str | None,
    resume: bool,
    force: bool,
    local_dir_use_symlinks: bool | str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    role: str,
    max_retries: int,
) -> dict[str, Any]:
    local_dir = repo_id_to_local_path(output_dir, local_layout_id)
    if local_dir.exists() and force:
        LOGGER.warning("Removing existing directory because --force was provided: %s", local_dir)
        shutil.rmtree(local_dir)

    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        validation = validate_downloaded_repo(local_dir, role)
        if validation["critical_ok"]:
            LOGGER.info("Local directory already exists and looks usable; skipping download: %s", local_dir)
            return write_download_record(
                local_dir=local_dir,
                backend=backend,
                source_id=source_id,
                local_layout_id=local_layout_id,
                revision=revision,
                skipped=True,
                validation=validation,
            )
        LOGGER.warning("Local directory exists but looks incomplete; attempting resumable download: %s", local_dir)
        for warning in validation["warnings"]:
            LOGGER.warning("%s", warning)

    LOGGER.info("Downloading %s from %s (%s) to %s", source_id, backend, revision, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    download_call = _download_call_for_backend(
        backend=backend,
        source_id=source_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        resume=resume,
        local_dir_use_symlinks=local_dir_use_symlinks,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    run_with_retries(download_call, backend=backend, source_id=source_id, local_dir=local_dir, max_retries=max_retries)

    validation = validate_downloaded_repo(local_dir, role)
    for warning in validation["warnings"]:
        LOGGER.warning("%s", warning)
    record = write_download_record(
        local_dir=local_dir,
        backend=backend,
        source_id=source_id,
        local_layout_id=local_layout_id,
        revision=revision,
        skipped=False,
        validation=validation,
    )
    if not validation["critical_ok"]:
        raise SystemExit(
            f"Downloaded files for {source_id} did not pass critical validation. "
            f"See warnings above and record file: {local_dir / 'download_record.json'}"
        )
    LOGGER.info("Ready: %s", record["local_path"])
    return record


def _download_call_for_backend(
    backend: str,
    source_id: str,
    local_dir: Path,
    revision: str,
    token: str | None,
    cache_dir: str | None,
    resume: bool,
    local_dir_use_symlinks: bool | str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> Callable[[], None]:
    if backend == "modelscope":
        return lambda: download_with_modelscope(
            model_id=source_id,
            local_dir=local_dir,
            revision=revision,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
        )
    return lambda: download_with_huggingface(
        repo_id=source_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        resume=resume,
        local_dir_use_symlinks=local_dir_use_symlinks,
    )


def download_with_modelscope(
    model_id: str,
    local_dir: str | Path,
    revision: str,
    ignore_patterns: list[str] | None = None,
    allow_patterns: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Download files with the ModelScope SDK without importing torch."""
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("modelscope is required. Install it with: pip install -U modelscope") from exc

    kwargs = _filter_supported_kwargs(
        snapshot_download,
        {
            "model_id": model_id,
            **_modelscope_revision_kwargs(snapshot_download, revision),
            "local_dir": str(local_dir),
            "cache_dir": str(cache_dir) if cache_dir else None,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        },
    )
    try:
        downloaded_path = snapshot_download(**kwargs)
    except TypeError:
        try:
            minimal_local = _filter_supported_kwargs(
                snapshot_download,
                {
                    "model_id": model_id,
                    **_modelscope_revision_kwargs(snapshot_download, revision),
                    "local_dir": str(local_dir),
                },
            )
            downloaded_path = snapshot_download(**minimal_local)
        except TypeError:
            fallback = _filter_supported_kwargs(
                snapshot_download,
                {
                    "model_id": model_id,
                    **_modelscope_revision_kwargs(snapshot_download, revision),
                    "cache_dir": str(local_dir),
                },
            )
            downloaded_path = snapshot_download(**fallback)
    sync_returned_snapshot_to_local_dir(downloaded_path, Path(local_dir))


def download_with_huggingface(
    repo_id: str,
    local_dir: str | Path,
    revision: str,
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
    allow_patterns: list[str] | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    local_dir_use_symlinks: bool | str = False,
) -> None:
    """Download files with Hugging Face Hub without importing torch."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install it with: pip install -U huggingface_hub") from exc

    kwargs = _filter_supported_kwargs(
        snapshot_download,
        {
            "repo_id": repo_id,
            "revision": revision,
            "local_dir": str(local_dir),
            "token": token,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "resume_download": resume,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
            "local_dir_use_symlinks": None if local_dir_use_symlinks == "auto" else local_dir_use_symlinks,
        },
    )
    snapshot_download(**kwargs)


def run_with_retries(
    download_call: Callable[[], None],
    backend: str,
    source_id: str,
    local_dir: Path,
    max_retries: int,
) -> None:
    attempts = max(1, max_retries)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            download_call()
            return
        except Exception as exc:
            last_exc = exc
            if backend == "huggingface" and is_auth_error(exc):
                raise SystemExit(
                    f"Failed to download {source_id}: authentication is required. "
                    "Please run huggingface-cli login or pass --hf-token."
                ) from exc
            if attempt < attempts:
                LOGGER.warning(
                    "Download attempt %s/%s failed for %s from %s: %s. Retrying with partial files kept in %s.",
                    attempt,
                    attempts,
                    source_id,
                    backend,
                    exc,
                    local_dir,
                )
                time.sleep(min(30, 2**attempt))
            else:
                break

    assert last_exc is not None
    if backend == "modelscope":
        raise SystemExit(
            f"Failed to download {source_id} from ModelScope after {attempts} attempt(s): {last_exc}\n"
            f"Partial files, if any, were kept in {local_dir} for resume.\n{MODELSCOPE_OVERRIDE_HINT}"
        ) from last_exc
    raise SystemExit(
        f"Failed to download {source_id} from Hugging Face after {attempts} attempt(s): {last_exc}\n"
        f"Partial files, if any, were kept in {local_dir} for resume."
    ) from last_exc


def validate_downloaded_repo(local_dir: Path, role: str) -> dict[str, Any]:
    warnings: list[str] = []
    critical_ok = True
    if not local_dir.exists():
        return {"critical_ok": False, "warnings": [f"Directory does not exist: {local_dir}"], "files_count": 0}

    files = [path for path in local_dir.rglob("*") if path.is_file()]
    if not files:
        return {"critical_ok": False, "warnings": [f"No files found in {local_dir}"], "files_count": 0}

    if not (local_dir / "config.json").exists():
        warnings.append(f"config.json was not found in {local_dir}")
        critical_ok = False

    if role == "main_model":
        if not has_any(local_dir, ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")):
            warnings.append(f"No tokenizer_config.json/tokenizer.json/tokenizer.model found in {local_dir}")
        if not has_weight_file(local_dir):
            warnings.append(f"No model weight shard found in {local_dir}")
            critical_ok = False
    elif role == "vision_tokenizer":
        if not has_weight_file(local_dir):
            warnings.append(f"No vision tokenizer weight shard found in {local_dir}")
            critical_ok = False

    return {"critical_ok": critical_ok, "warnings": warnings, "files_count": len(files)}


def write_download_record(
    local_dir: Path,
    backend: str,
    source_id: str,
    local_layout_id: str,
    revision: str,
    skipped: bool,
    validation: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "backend": backend,
        "repo_id": source_id,
        "source_id": source_id,
        "local_layout_id": local_layout_id,
        "local_path": str(local_dir),
        "revision": revision,
        "download_time": datetime.now(timezone.utc).isoformat(),
        "downloaded_files_count": validation["files_count"],
        "skipped_existing": skipped,
        "validation": validation,
    }
    (local_dir / "download_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _filter_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in kwargs.items() if value is not None}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return clean
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return clean
    return {key: value for key, value in clean.items() if key in sig.parameters}


def _modelscope_revision_kwargs(fn: Callable[..., Any], revision: str) -> dict[str, str]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {"revision": revision}
    if "revision" in sig.parameters:
        return {"revision": revision}
    if "model_revision" in sig.parameters:
        return {"model_revision": revision}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return {"revision": revision}
    return {}


def sync_returned_snapshot_to_local_dir(downloaded_path: Any, local_dir: Path) -> None:
    if not downloaded_path:
        return
    source = Path(str(downloaded_path)).expanduser()
    target = local_dir.expanduser()
    try:
        if source.resolve() == target.resolve():
            return
    except OSError:
        return
    if not source.exists() or not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        destination = target / child.name
        try:
            if child.resolve() == destination.resolve():
                continue
        except OSError:
            pass
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def has_weight_file(path: Path) -> bool:
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.index.json")
    return any(next(path.rglob(pattern), None) is not None for pattern in patterns)


def has_any(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).exists() for name in names)


def is_auth_error(exc: Exception) -> bool:
    message = str(exc)
    return "401" in message or "403" in message or "Unauthorized" in message or "gated" in message.lower()


def parse_bool(value: str) -> bool | str:
    if value == "auto":
        return "auto"
    return value.lower() == "true"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


if __name__ == "__main__":
    main()
