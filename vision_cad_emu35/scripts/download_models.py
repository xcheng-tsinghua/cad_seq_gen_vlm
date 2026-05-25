from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.model_paths import (
    DEFAULT_MAIN_REPO_ID,
    DEFAULT_MODEL_ROOT,
    DEFAULT_VISION_TOKENIZER_REPO_ID,
    model_config_to_yaml_snippet,
    repo_id_to_local_path,
)


LOGGER = logging.getLogger("download_models")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install it with: pip install -U huggingface_hub") from exc

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    downloaded: list[dict[str, Any]] = []

    if not args.skip_main_model:
        downloaded.append(
            download_repo(
                snapshot_download=snapshot_download,
                repo_id=args.main_repo_id,
                output_dir=output_dir,
                revision=args.revision,
                token=token,
                cache_dir=args.cache_dir,
                resume=args.resume,
                force=args.force,
                local_dir_use_symlinks=parse_bool(args.local_dir_use_symlinks),
                include_patterns=args.include_pattern,
                exclude_patterns=args.exclude_pattern,
                role="main_model",
            )
        )

    if not args.skip_vision_tokenizer:
        downloaded.append(
            download_repo(
                snapshot_download=snapshot_download,
                repo_id=args.vision_tokenizer_repo_id,
                output_dir=output_dir,
                revision=args.revision,
                token=token,
                cache_dir=args.cache_dir,
                resume=args.resume,
                force=args.force,
                local_dir_use_symlinks=parse_bool(args.local_dir_use_symlinks),
                include_patterns=args.include_pattern,
                exclude_patterns=args.exclude_pattern,
                role="vision_tokenizer",
            )
        )

    print()
    print("Downloaded/local model paths:")
    for item in downloaded:
        print(f"- {item['repo_id']} -> {item['local_path']}")
    print()
    print("Pasteable runtime config snippet:")
    print(model_config_to_yaml_snippet(output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only Hugging Face downloader for Emu3.5 weights/tokenizers. This script does not import torch."
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Falls back to HF_TOKEN or login cache.")
    parser.add_argument("--main-repo-id", default=DEFAULT_MAIN_REPO_ID)
    parser.add_argument("--vision-tokenizer-repo-id", default=DEFAULT_VISION_TOKENIZER_REPO_ID)
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Delete an existing local repo directory and download again.")
    parser.add_argument("--local-dir-use-symlinks", default="false", choices=["true", "false", "auto"])
    parser.add_argument("--include-pattern", action="append", default=None)
    parser.add_argument("--exclude-pattern", action="append", default=None)
    parser.add_argument("--skip-main-model", action="store_true")
    parser.add_argument("--skip-vision-tokenizer", action="store_true")
    return parser


def download_repo(
    snapshot_download: Any,
    repo_id: str,
    output_dir: Path,
    revision: str,
    token: str | None,
    cache_dir: str | None,
    resume: bool,
    force: bool,
    local_dir_use_symlinks: bool | str,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    role: str,
) -> dict[str, Any]:
    local_dir = repo_id_to_local_path(output_dir, repo_id)
    if local_dir.exists() and force:
        LOGGER.warning("Removing existing directory because --force was provided: %s", local_dir)
        shutil.rmtree(local_dir)

    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        validation = validate_downloaded_repo(local_dir, role)
        if validation["critical_ok"]:
            LOGGER.info("Local directory already exists and looks usable; skipping download: %s", local_dir)
            record = write_download_record(local_dir, repo_id, revision, skipped=True, validation=validation)
            return record
        LOGGER.warning("Local directory exists but looks incomplete; attempting resumable download: %s", local_dir)
        for warning in validation["warnings"]:
            LOGGER.warning("%s", warning)

    LOGGER.info("Downloading %s (%s) to %s", repo_id, revision, local_dir)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": str(local_dir),
        "token": token,
        "resume_download": resume,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if include_patterns:
        kwargs["allow_patterns"] = include_patterns
    if exclude_patterns:
        kwargs["ignore_patterns"] = exclude_patterns
    if local_dir_use_symlinks != "auto":
        kwargs["local_dir_use_symlinks"] = local_dir_use_symlinks

    try:
        snapshot_download(**kwargs)
    except Exception as exc:
        message = str(exc)
        if "401" in message or "403" in message or "Unauthorized" in message or "gated" in message.lower():
            raise SystemExit(
                f"Failed to download {repo_id}: authentication is required. "
                "Please run huggingface-cli login or pass --hf-token."
            ) from exc
        raise SystemExit(
            f"Failed to download {repo_id}. Network or Hugging Face Hub error: {exc}\n"
            f"Partial files, if any, were kept in {local_dir} for resume."
        ) from exc

    validation = validate_downloaded_repo(local_dir, role)
    for warning in validation["warnings"]:
        LOGGER.warning("%s", warning)
    record = write_download_record(local_dir, repo_id, revision, skipped=False, validation=validation)
    if not validation["critical_ok"]:
        raise SystemExit(
            f"Downloaded files for {repo_id} did not pass critical validation. "
            f"See warnings above and record file: {local_dir / 'download_record.json'}"
        )
    LOGGER.info("Ready: %s", record["local_path"])
    return record


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
    repo_id: str,
    revision: str,
    skipped: bool,
    validation: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "repo_id": repo_id,
        "local_path": str(local_dir),
        "revision": revision,
        "download_time": datetime.now(timezone.utc).isoformat(),
        "downloaded_files_count": validation["files_count"],
        "skipped_existing": skipped,
        "validation": validation,
    }
    (local_dir / "download_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def has_weight_file(path: Path) -> bool:
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.index.json")
    return any(next(path.rglob(pattern), None) is not None for pattern in patterns)


def has_any(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).exists() for name in names)


def parse_bool(value: str) -> bool | str:
    if value == "auto":
        return "auto"
    return value.lower() == "true"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


if __name__ == "__main__":
    main()
