from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vision_cad_emu35.config import AppConfig
from vision_cad_emu35.model_paths import ensure_default_local_model_paths, validate_local_model_paths
from vision_cad_emu35.models.emu35_adapter import Emu35Adapter
from vision_cad_emu35.utils.image_io import load_image_rgb, save_image


def load_adapter_for_inference(config: AppConfig, checkpoint: str | Path | None = None) -> Emu35Adapter:
    ensure_default_local_model_paths(config.model)
    validate_local_model_paths(config.model)
    adapter = Emu35Adapter(config.model)
    adapter.load_model()
    if checkpoint:
        adapter.load_checkpoint(checkpoint)
    return adapter


def run_single_step(
    adapter: Emu35Adapter,
    final_snapshot_path: str | Path,
    prev_depth_map_path: str | Path,
    output_dir: str | Path,
    prompt: str | None,
    generation_config: Any,
) -> dict[str, Any]:
    start = time.perf_counter()
    final_image = load_image_rgb(final_snapshot_path)
    prev_image = load_image_rgb(prev_depth_map_path)
    result = adapter.generate(final_image, prev_image, prompt, generation_config)
    latency = time.perf_counter() - start

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "operation_type.txt").write_text(result["operation_type"], encoding="utf-8")
    save_image(result["image"], out / "overlayed_all.png")
    response = {
        "operation_type": result["operation_type"],
        "raw_text": result.get("raw_text", ""),
        "latency_seconds": latency,
        "metadata": result.get("metadata", {}),
    }
    (out / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    return {**result, "latency_seconds": latency, "output_dir": str(out)}
