from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import typer

from utils.image_ops import parse_roll_back_index

app = typer.Typer(add_completion=False)


def _collect_steps(part_dir: Path) -> List[Path]:
    step_dirs = [
        p for p in part_dir.iterdir() if p.is_dir() and p.name.startswith("roll_back_index_")
    ]
    step_dirs.sort(key=lambda p: parse_roll_back_index(p.name))
    return step_dirs


def _step_paths(step_dir: Path) -> Dict[str, str]:
    return {
        "prev_depth_map": str((step_dir / "prev_depth_map.png").as_posix()),
        "sketch_plane_mask": str((step_dir / "sketch_plane_mask.png").as_posix()),
        "reference_mask": str((step_dir / "reference_mask.png").as_posix()),
        "result_frame": str((step_dir / "result_frame.png").as_posix()),
    }


@app.command()
def main(
    raw_root: Path = typer.Option(..., help="Raw dataset root."),
    out_root: Path = typer.Option(..., help="Processed dataset root."),
    val_ratio: float = typer.Option(0.1, help="Validation ratio by part."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    random.seed(seed)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    part_dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    part_dirs.sort(key=lambda p: p.name)
    random.shuffle(part_dirs)

    val_count = max(1, int(len(part_dirs) * val_ratio)) if part_dirs else 0
    val_part_ids = {p.name for p in part_dirs[:val_count]}

    train_ids: List[str] = []
    val_ids: List[str] = []
    step_stats: Dict[str, int] = {}

    with manifest_path.open("w", encoding="utf-8") as fout:
        for part_dir in sorted(part_dirs, key=lambda x: x.name):
            step_dirs = _collect_steps(part_dir)
            if not step_dirs:
                continue

            part_id = part_dir.name
            step_stats[part_id] = len(step_dirs)
            final_result_frame = str((step_dirs[-1] / "result_frame.png").as_posix())

            for seq_idx, step_dir in enumerate(step_dirs, start=1):
                row = {
                    "sample_id": f"{part_id}__step_{seq_idx:03d}",
                    "part_id": part_id,
                    "step_index": seq_idx,
                    "split": "val" if part_id in val_part_ids else "train",
                    "target_part_image": final_result_frame,
                    "target": _step_paths(step_dir),
                    "prev_target": _step_paths(step_dirs[seq_idx - 2]) if seq_idx > 1 else None,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                if part_id in val_part_ids:
                    val_ids.append(row["sample_id"])
                else:
                    train_ids.append(row["sample_id"])

    (out_root / "train_split.json").write_text(
        json.dumps({"sample_ids": train_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "val_split.json").write_text(
        json.dumps({"sample_ids": val_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "step_stats.json").write_text(
        json.dumps(step_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(
        f"Done. parts={len(step_stats)}, train_samples={len(train_ids)}, val_samples={len(val_ids)}"
    )


if __name__ == "__main__":
    app()

