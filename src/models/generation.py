from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from config import GenerationConfig
from models.emu35_adapter import Emu35Adapter
from utils.image_io import save_image


def generate_multimodal(
    adapter: Emu35Adapter,
    prompt_text: str,
    images: list[Image.Image],
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    return adapter.generate_multimodal(prompt_text, images, generation_config)


def save_generation_result(result: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "operation_type.txt").write_text(result.get("operation_type", ""), encoding="utf-8")
    if result.get("image") is not None:
        save_image(result["image"], out / "overlayed_all.png")
    response = {k: v for k, v in result.items() if k != "image"}
    (out / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
