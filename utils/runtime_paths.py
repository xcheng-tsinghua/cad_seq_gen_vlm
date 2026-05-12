from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_name_from_path(path: Path) -> str:
    name = path.resolve().name.strip()
    return name if name else "dataset"


def dataset_output_root(raw_root: Path) -> Path:
    return project_root() / "output" / dataset_name_from_path(raw_root)


def dataset_model_root(raw_root: Path) -> Path:
    return project_root() / "model_trained" / dataset_name_from_path(raw_root)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def auto_run_dir(raw_root: Path, mode: str) -> Path:
    base = dataset_model_root(raw_root) if mode == "train" else dataset_output_root(raw_root)
    out = base / f"{mode}_{timestamp_tag()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def latest_checkpoint_marker(raw_root: Path) -> Path:
    return dataset_model_root(raw_root) / "latest_best_checkpoint.txt"


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
    roots = [dataset_model_root(raw_root), dataset_output_root(raw_root)]
    cands: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        cands.extend(root.rglob("best.pth"))
        # Backward compatibility for old naming.
        cands.extend(root.rglob("best.pt"))
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

