#!/usr/bin/env python3
"""Verify and download HuggingFace model weights into ``pretrained_lm/``.

This project pins the HuggingFace cache to :data:`config.PRETRAINED_DIR` on
import (``HF_HOME`` / ``HF_HUB_CACHE`` / etc.). Run this script once per
machine (or after a partial download) to ensure every checkpoint that
``train.py``, ``inference.py``, and ``auto_label.py`` need is present and
consistent with the remote revision.

Default repos (edit ``config.ModelConfig`` / ``auto_label.LabelerConfig`` if
you change them in code):

* ``stabilityai/stable-diffusion-xl-base-1.0`` — SDXL backbone (diffusers)
* ``openai/clip-vit-large-patch14`` — CLIP vision for IP-Adapter
* ``Qwen/Qwen2.5-VL-7B-Instruct`` — Qwen2.5-VL for pseudo-labeling

Examples::

    python download_pretrained.py
    python download_pretrained.py --check-only
    python download_pretrained.py --skip-qwen
    python download_pretrained.py --qwen-model Qwen/Qwen2.5-VL-3B-Instruct
    python download_pretrained.py --token $HF_TOKEN

Requires: ``pip install huggingface_hub`` (installed with ``transformers``).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Must run first so HF_* env vars point at pretrained_lm before hub I/O.
import config  # noqa: F401

from config import PRETRAINED_DIR, ModelConfig

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. Install with:\n"
        "  pip install -U huggingface_hub\n"
        "or install project requirements.txt (includes transformers)."
    ) from exc


@dataclass(frozen=True)
class RepoTask:
    """One HF repo to snapshot into ``PRETRAINED_DIR``."""

    key: str
    repo_id: str
    repo_type: str = "model"
    revision: Optional[str] = None


def _snapshot(
    task: RepoTask,
    *,
    token: Optional[str],
    local_files_only: bool,
) -> str:
    kwargs = dict(
        repo_id=task.repo_id,
        repo_type=task.repo_type,
        cache_dir=PRETRAINED_DIR,
        token=token,
        resume_download=True,
        max_workers=4,
    )
    if task.revision:
        kwargs["revision"] = task.revision
    if local_files_only:
        kwargs["local_files_only"] = True
    return snapshot_download(**kwargs)


def _run_tasks(
    tasks: Sequence[RepoTask],
    *,
    check_only: bool,
    token: Optional[str],
) -> bool:
    """Return True if every repo is present (check-only) or downloaded successfully."""
    mode = "verify local cache only" if check_only else "download / resume"
    print(f"HF cache directory: {PRETRAINED_DIR}")
    print(f"Mode: {mode}\n")

    for task in tasks:
        print(f"[{task.key}] {task.repo_id}")
        try:
            path = _snapshot(task, token=token, local_files_only=check_only)
        except LocalEntryNotFoundError:
            print(
                f"  FAIL: incomplete or missing in cache. "
                f"Run again without --check-only to download.\n"
            )
            return False
        except FileNotFoundError as exc:
            # Older huggingface_hub builds raise this (e.g. missing refs/main)
            # when the repo has never been downloaded.
            print(
                f"  FAIL: nothing in cache for this repo ({exc}). "
                f"Run again without --check-only to download.\n"
            )
            return False
        except RepositoryNotFoundError as exc:
            print(f"  FAIL: repository not found ({exc}).\n")
            return False
        except HfHubHTTPError as exc:
            print(f"  FAIL: HTTP error — check network / token / license ({exc}).\n")
            return False

        print(f"  OK: {path}\n")

    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download or verify HuggingFace weights under pretrained_lm/.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Do not hit the network; fail if any file is missing from cache.",
    )
    p.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip Qwen2.5-VL (only fetch SDXL + CLIP, for train/infer-only setups).",
    )
    p.add_argument(
        "--skip-diffusion",
        action="store_true",
        help="Skip SDXL + CLIP (only fetch Qwen, for auto_label-only setups).",
    )
    p.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HF repo id for Qwen2.5-VL (default matches auto_label.py).",
    )
    p.add_argument(
        "--sdxl-repo",
        type=str,
        default=None,
        help=f"Override SDXL repo id (default: {ModelConfig.pretrained_model_name_or_path!r}).",
    )
    p.add_argument(
        "--clip-repo",
        type=str,
        default=None,
        help=f"Override CLIP vision repo id (default: {ModelConfig.clip_image_encoder_name_or_path!r}).",
    )
    p.add_argument(
        "--revision-sdxl",
        type=str,
        default=None,
        help="Optional Git revision (branch / tag / commit) for SDXL repo.",
    )
    p.add_argument(
        "--revision-clip",
        type=str,
        default=None,
        help="Optional Git revision for CLIP repo.",
    )
    p.add_argument(
        "--revision-qwen",
        type=str,
        default=None,
        help="Optional Git revision for Qwen repo.",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var). Needed for gated models.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    mc = ModelConfig()
    sdxl_id = args.sdxl_repo or mc.pretrained_model_name_or_path
    clip_id = args.clip_repo or mc.clip_image_encoder_name_or_path

    tasks: List[RepoTask] = []
    if not args.skip_diffusion:
        tasks.append(
            RepoTask(
                key="sdxl",
                repo_id=sdxl_id,
                revision=args.revision_sdxl,
            )
        )
        tasks.append(
            RepoTask(
                key="clip_vision",
                repo_id=clip_id,
                revision=args.revision_clip,
            )
        )
    if not args.skip_qwen:
        tasks.append(
            RepoTask(
                key="qwen25_vl",
                repo_id=args.qwen_model,
                revision=args.revision_qwen,
            )
        )

    if not tasks:
        print("Nothing to do (--skip-qwen and --skip-diffusion both set).")
        sys.exit(0)

    if not _run_tasks(tasks, check_only=args.check_only, token=token):
        sys.exit(1)

    if args.check_only:
        print("All selected repositories are complete in the local HF cache.")
    else:
        print("All selected repositories are ready under:", PRETRAINED_DIR)


if __name__ == "__main__":
    main()
