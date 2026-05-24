from __future__ import annotations

import shutil
from pathlib import Path


def rotate_checkpoints(output_dir: str | Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    root = Path(output_dir)
    checkpoints = sorted(
        [p for p in root.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    while len(checkpoints) > save_total_limit:
        victim = checkpoints.pop(0)
        shutil.rmtree(victim)


def copy_best_checkpoint(source: str | Path, output_dir: str | Path) -> Path:
    src = Path(source)
    dst = Path(output_dir) / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst

