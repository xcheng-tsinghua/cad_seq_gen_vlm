from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
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


def make_step_canvas(step_images: Dict[str, Image.Image], panel_size: int) -> Image.Image:
    canvas = Image.new("RGB", (panel_size * 2, panel_size * 2), color=(0, 0, 0))
    canvas.paste(to_rgb(step_images["prev_depth_map"]).resize((panel_size, panel_size)), (0, 0))
    canvas.paste(to_rgb(step_images["sketch_plane_mask"]).resize((panel_size, panel_size)), (panel_size, 0))
    canvas.paste(to_rgb(step_images["reference_mask"]).resize((panel_size, panel_size)), (0, panel_size))
    canvas.paste(to_rgb(step_images["result_frame"]).resize((panel_size, panel_size)), (panel_size, panel_size))
    return canvas


def split_step_canvas(canvas: Image.Image) -> Dict[str, Image.Image]:
    w, h = canvas.size
    if w % 2 != 0 or h % 2 != 0:
        raise ValueError(f"Canvas shape must be even, got: {(w, h)}")
    panel_w, panel_h = w // 2, h // 2
    return {
        "prev_depth_map": canvas.crop((0, 0, panel_w, panel_h)),
        "sketch_plane_mask": canvas.crop((panel_w, 0, w, panel_h)),
        "reference_mask": canvas.crop((0, panel_h, panel_w, h)),
        "result_frame": canvas.crop((panel_w, panel_h, w, h)),
    }


def make_condition_canvas(
    part_image: Image.Image,
    prev_canvas: Image.Image | None,
    panel_size: int,
) -> Image.Image:
    """Build ControlNet condition image.

    Layout:
    - TL: target part image (global objective)
    - TR: previous generated step canvas (autoregressive context)
    - BL: Canny edge of target part image (shape prior)
    - BR: blank
    """
    part = to_rgb(part_image).resize((panel_size, panel_size))
    prev = (
        to_rgb(prev_canvas).resize((panel_size, panel_size))
        if prev_canvas is not None
        else Image.new("RGB", (panel_size, panel_size), color=(0, 0, 0))
    )
    edges = _canny_rgb(part)
    blank = Image.new("RGB", (panel_size, panel_size), color=(0, 0, 0))

    canvas = Image.new("RGB", (panel_size * 2, panel_size * 2), color=(0, 0, 0))
    canvas.paste(part, (0, 0))
    canvas.paste(prev, (panel_size, 0))
    canvas.paste(edges, (0, panel_size))
    canvas.paste(blank, (panel_size, panel_size))
    return canvas


def _canny_rgb(img: Image.Image, low: int = 80, high: int = 180) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, threshold1=low, threshold2=high)
    edge_rgb = np.stack([edge, edge, edge], axis=-1)
    return Image.fromarray(edge_rgb)


def save_step_images(step_images: Dict[str, Image.Image], step_dir: Path) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    for key in STEP_KEYS:
        step_images[key].save(step_dir / f"{key}.png")


def parse_roll_back_index(name: str) -> int:
    prefix = "roll_back_index_"
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected folder name: {name}")
    return int(name[len(prefix) :])


def pil_to_torch(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def torch_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.transpose(arr, (1, 2, 0))
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def ensure_rgba_removed(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def image_size_hw(img: Image.Image) -> Tuple[int, int]:
    return img.size[1], img.size[0]

