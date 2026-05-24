from __future__ import annotations

from pathlib import Path

from PIL import Image

from vision_cad_emu35.utils.image_io import blank_rgb, load_and_resize, resize_pad_image


def load_preprocessed_image(path: str | Path, image_size: int = 512, allow_blank: bool = False) -> Image.Image:
    if allow_blank and not str(path):
        return blank_rgb(image_size)
    return load_and_resize(path, size=image_size)


def preprocess_pil_image(image: Image.Image, image_size: int = 512) -> Image.Image:
    return resize_pad_image(image, size=image_size)

