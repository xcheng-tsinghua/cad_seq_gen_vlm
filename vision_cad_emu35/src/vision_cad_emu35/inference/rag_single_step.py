from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vision_cad_emu35.config import AppConfig
from vision_cad_emu35.model_paths import apply_model_root_override, ensure_default_local_model_paths, validate_local_model_paths
from vision_cad_emu35.models.emu35_adapter import Emu35Adapter
from vision_cad_emu35.rag.prompt_builder import RagPromptBuilder
from vision_cad_emu35.rag.retriever import RagRetriever
from vision_cad_emu35.utils.image_io import load_image_rgb, save_image


def load_frozen_adapter(config: AppConfig) -> Emu35Adapter:
    ensure_default_local_model_paths(config.model)
    validate_local_model_paths(config.model)
    adapter = Emu35Adapter(config.model)
    adapter.load_model()
    return adapter


def run_rag_single_step(
    config: AppConfig,
    adapter: Emu35Adapter,
    retriever: RagRetriever,
    final_snapshot_path: str | Path,
    prev_depth_map_path: str | Path,
    output_dir: str | Path,
    top_k: int | None = None,
    prompt_extra: str | None = None,
    operation_type_hint: str | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    final_image = load_image_rgb(final_snapshot_path)
    prev_image = load_image_rgb(prev_depth_map_path)
    filters = {"operation_type_hint": operation_type_hint} if operation_type_hint else None
    retrieved = retriever.retrieve(
        final_image,
        prev_image,
        top_k=top_k or config.rag.top_k,
        filters=filters,
    )
    prompt = RagPromptBuilder(config.rag, image_size=config.model.image_size).build(
        final_image,
        prev_image,
        retrieved,
        prompt_extra=prompt_extra,
        operation_type_hint=operation_type_hint,
    )
    result = adapter.generate_multimodal(prompt.prompt_text, prompt.images, config.generation)
    latency = time.perf_counter() - start
    zero_shot = len(retrieved) == 0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "operation_type.txt").write_text(result.get("operation_type", ""), encoding="utf-8")
    (out / "prompt.txt").write_text(prompt.prompt_text, encoding="utf-8")
    (out / "retrieved_examples.json").write_text(json.dumps(prompt.retrieved_examples, indent=2, default=str), encoding="utf-8")
    if result.get("image") is not None:
        save_image(result["image"], out / "overlayed_all.png")
    response = {
        "operation_type": result.get("operation_type"),
        "raw_text": result.get("raw_text", ""),
        "metadata": result.get("metadata", {}),
        "latency_seconds": latency,
        "zero_shot": zero_shot,
        "warning": "No retrieved examples available; running zero-shot mode." if zero_shot else None,
        "num_retrieved": len(retrieved),
        "image_roles": prompt.image_roles,
    }
    (out / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    return {**response, "image": result.get("image"), "retrieved_examples": prompt.retrieved_examples, "output_dir": str(out)}
