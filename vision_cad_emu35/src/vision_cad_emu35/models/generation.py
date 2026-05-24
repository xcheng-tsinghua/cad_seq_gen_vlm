from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from vision_cad_emu35.config import GenerationConfig
from vision_cad_emu35.models.emu35_adapter import Emu35Adapter


def generate_step(
    adapter: Emu35Adapter,
    final_snapshot: Image.Image,
    prev_depth_map: Image.Image,
    prompt: str | None,
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    return adapter.generate(final_snapshot, prev_depth_map, prompt, generation_config)


def save_generation_result(result: dict[str, Any], output_dir: str | Path) -> None:
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "operation_type.txt").write_text(result["operation_type"], encoding="utf-8")
    result["image"].save(out / "overlayed_all.png")
    response = {k: v for k, v in result.items() if k != "image"}
    (out / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")

