from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_image_rgb(path: str | Path) -> Image.Image:
    """Load an image with PIL and convert it to RGB."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    with Image.open(image_path) as im:
        return im.convert("RGB")


def resize_pad_image(
    image: Image.Image,
    size: int = 512,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize with aspect ratio preserved and pad to a square canvas."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def load_and_resize(path: str | Path, size: int = 512) -> Image.Image:
    return resize_pad_image(load_image_rgb(path), size=size)


def save_image(image: Image.Image, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_to_image(data: str) -> Image.Image:
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def validate_image_file(path: str | Path) -> bool:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return False
    try:
        with Image.open(image_path) as im:
            im.verify()
        return True
    except Exception:
        return False


def blank_rgb(size: int = 512, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    return Image.new("RGB", (size, size), fill)

