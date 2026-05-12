from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import typer
from PIL import Image

from src.cad_seq_gen.utils.image_ops import (
    STEP_KEYS,
    make_condition_canvas,
    make_step_canvas,
    parse_roll_back_index,
    read_gray,
)

app = typer.Typer(add_completion=False)


def _collect_steps(part_dir: Path) -> List[Path]:
    step_dirs = [
        p for p in part_dir.iterdir() if p.is_dir() and p.name.startswith("roll_back_index_")
    ]
    step_dirs.sort(key=lambda p: parse_roll_back_index(p.name))
    return step_dirs


def _load_step_images(step_dir: Path, size: int) -> Dict[str, Image.Image]:
    return {
        "prev_depth_map": read_gray(step_dir / "prev_depth_map.png", size=size).convert("RGB"),
        "sketch_plane_mask": read_gray(step_dir / "sketch_plane_mask.png", size=size).convert("RGB"),
        "reference_mask": read_gray(step_dir / "reference_mask.png", size=size).convert("RGB"),
        "result_frame": read_gray(step_dir / "result_frame.png", size=size).convert("RGB"),
    }


@app.command()
def main(
    raw_root: Path = typer.Option(..., help="Raw dataset root."),
    out_root: Path = typer.Option(..., help="Processed dataset root."),
    image_size: int = typer.Option(512, help="Single panel size."),
    val_ratio: float = typer.Option(0.1, help="Validation ratio by part."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    random.seed(seed)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "targets").mkdir(exist_ok=True)
    (out_root / "controls").mkdir(exist_ok=True)

    part_dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    part_dirs.sort(key=lambda p: p.name)
    random.shuffle(part_dirs)
    val_count = max(1, int(len(part_dirs) * val_ratio)) if part_dirs else 0
    val_part_ids = {p.name for p in part_dirs[:val_count]}

    manifest_path = out_root / "manifest.jsonl"
    train_ids, val_ids = [], []
    step_stats = {}

    with manifest_path.open("w", encoding="utf-8") as fout:
        for part_dir in sorted(part_dirs, key=lambda x: x.name):
            step_dirs = _collect_steps(part_dir)
            if not step_dirs:
                continue

            part_id = part_dir.name
            step_stats[part_id] = len(step_dirs)
            final_step = step_dirs[-1]
            target_part_img = read_gray(final_step / "result_frame.png", size=image_size).convert("RGB")

            prev_canvas = None
            for seq_idx, step_dir in enumerate(step_dirs, start=1):
                step_imgs = _load_step_images(step_dir, size=image_size)
                target_canvas = make_step_canvas(step_imgs, panel_size=image_size)
                cond_canvas = make_condition_canvas(
                    part_image=target_part_img,
                    prev_canvas=prev_canvas,
                    panel_size=image_size,
                )

                sample_id = f"{part_id}__step_{seq_idx:03d}"
                target_path = out_root / "targets" / f"{sample_id}.png"
                control_path = out_root / "controls" / f"{sample_id}.png"
                target_canvas.save(target_path)
                cond_canvas.save(control_path)

                row = {
                    "sample_id": sample_id,
                    "part_id": part_id,
                    "step_index": seq_idx,
                    "prompt": "cad modeling step canvas",
                    "target_image": str(target_path.as_posix()),
                    "control_image": str(control_path.as_posix()),
                    "split": "val" if part_id in val_part_ids else "train",
                    "keys": list(STEP_KEYS),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

                if part_id in val_part_ids:
                    val_ids.append(sample_id)
                else:
                    train_ids.append(sample_id)
                prev_canvas = target_canvas

    (out_root / "train_split.json").write_text(
        json.dumps({"sample_ids": train_ids}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "val_split.json").write_text(
        json.dumps({"sample_ids": val_ids}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "step_stats.json").write_text(
        json.dumps(step_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    typer.echo(
        f"Done. parts={len(step_stats)}, train_samples={len(train_ids)}, val_samples={len(val_ids)}"
    )


if __name__ == "__main__":
    app()

