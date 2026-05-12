from __future__ import annotations

from pathlib import Path
from typing import Dict

from PIL import Image


STEP_KEYS = (
    "prev_depth_map",
    "sketch_plane_mask",
    "reference_mask",
    "result_frame",
)


def read_rgb(path: Path, size: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return img


def read_gray(path: Path, size: int | None = None) -> Image.Image:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return img


def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def save_step_images(step_images: Dict[str, Image.Image], step_dir: Path) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    for key in STEP_KEYS:
        step_images[key].save(step_dir / f"{key}.png")


def parse_roll_back_index(name: str) -> int:
    prefix = "roll_back_index_"
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected folder name: {name}")
    return int(name[len(prefix) :])

