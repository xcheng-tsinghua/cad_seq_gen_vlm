from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config import AppConfig
from model_paths import apply_model_root_override, ensure_default_local_model_paths, validate_local_model_paths
from models.emu35_adapter import Emu35Adapter
from rag.prompt_builder import RagPromptBuilder
from rag.retriever import RagRetriever
from utils.image_io import load_image_rgb, save_image
from utils.runtime_env import normalize_thread_env


def load_frozen_adapter(config: AppConfig) -> Emu35Adapter:
    normalize_thread_env()
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
    operation_type = adapter.parse_operation_type(result.get("raw_text", ""))
    latency = time.perf_counter() - start
    zero_shot = len(retrieved) == 0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "operation_type.txt").write_text(operation_type, encoding="utf-8")
    (out / "prompt.txt").write_text(prompt.prompt_text, encoding="utf-8")
    (out / "retrieved_examples.json").write_text(json.dumps(prompt.retrieved_examples, indent=2, default=str), encoding="utf-8")

    generated_images = list(result.get("images") or [])
    if not generated_images and result.get("image") is not None:
        generated_images = [result["image"]]
    generated_image_paths: list[str] = []
    image_path: str | None = None
    if generated_images:
        overlay_path = out / "overlayed_all.png"
        save_image(generated_images[0], overlay_path)
        image_path = str(overlay_path)
        generated_dir = out / "generated_images"
        for index, image in enumerate(generated_images):
            target = generated_dir / f"image_{index:03d}.png"
            save_image(image, target)
            generated_image_paths.append(str(target))

    debug_events = result.get("emu35_events_debug") or result.get("metadata", {}).get("event_summaries") or []
    debug_events_path: str | None = None
    if getattr(config.generation, "save_debug_events", True) or not generated_images:
        debug_path = out / "emu35_events_debug.json"
        debug_payload = {
            "num_generation_events": result.get("metadata", {}).get("num_generation_events", len(debug_events)),
            "num_generated_images": len(generated_images),
            "raw_text_missing": result.get("raw_text_missing", not bool(result.get("raw_text", ""))),
            "events": debug_events,
        }
        debug_path.write_text(json.dumps(debug_payload, indent=2, default=str), encoding="utf-8")
        debug_events_path = str(debug_path)

    response = {
        "operation_type": operation_type,
        "raw_text": result.get("raw_text", ""),
        "raw_text_missing": result.get("raw_text_missing", not bool(result.get("raw_text", ""))),
        "metadata": result.get("metadata", {}),
        "latency_seconds": latency,
        "zero_shot": zero_shot,
        "warning": "No retrieved examples available; running zero-shot mode." if zero_shot else None,
        "kb_dir": str(retriever.kb_dir),
        "num_retrieved": len(retrieved),
        "image_roles": prompt.image_roles,
        "image_path": image_path,
        "num_generated_images": len(generated_images),
        "generated_image_paths": generated_image_paths,
        "debug_events_path": debug_events_path,
        "image_missing": not bool(generated_images),
    }
    if not generated_images:
        response.update(
            {
                "image_missing_reason": "No PIL image was found in Emu3.5 generation events.",
            }
        )
    (out / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    if generated_images:
        print(f"[INFO] Saved generated preview image: {image_path}")
    else:
        print(f"[WARNING] No generated image found in Emu3.5 output. Debug events saved to: {debug_events_path}")
    return {**response, "image": generated_images[0] if generated_images else None, "retrieved_examples": prompt.retrieved_examples, "output_dir": str(out)}
