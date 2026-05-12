from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dataset_name_from_path(path: Path) -> str:
    name = path.resolve().name.strip()
    return name if name else "dataset"


def dataset_output_root(raw_root: Path) -> Path:
    return project_root() / "outputs" / dataset_name_from_path(raw_root)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def auto_run_dir(raw_root: Path, mode: str) -> Path:
    out = dataset_output_root(raw_root) / f"{mode}_{timestamp_tag()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def latest_checkpoint_marker(raw_root: Path) -> Path:
    return dataset_output_root(raw_root) / "latest_best_checkpoint.txt"


def save_latest_checkpoint(raw_root: Path, checkpoint: Path) -> None:
    marker = latest_checkpoint_marker(raw_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(checkpoint.resolve().as_posix()), encoding="utf-8")


def discover_latest_checkpoint(raw_root: Path) -> Path | None:
    marker = latest_checkpoint_marker(raw_root)
    if marker.exists():
        text = marker.read_text(encoding="utf-8").strip()
        if text:
            p = Path(text)
            if p.exists():
                return p
    root = dataset_output_root(raw_root)
    if not root.exists():
        return None
    cands = sorted(root.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

