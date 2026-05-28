from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from config import AppConfig
from model_paths import ensure_default_local_model_paths, validate_local_model_paths
from models.emu35_adapter import Emu35Adapter
from utils.image_io import load_image_rgb, resize_pad_image, save_image
from utils.runtime_env import normalize_thread_env


MAX_GENERAL_INPUT_IMAGES = 5


def load_general_adapter(config: AppConfig) -> Emu35Adapter:
    normalize_thread_env()
    ensure_default_local_model_paths(config.model)
    validate_local_model_paths(config.model)
    adapter = Emu35Adapter(config.model)
    adapter.load_model()
    return adapter


def load_general_images(
    image_paths: Iterable[str | Path] | None,
    image_size: int,
    max_images: int = MAX_GENERAL_INPUT_IMAGES,
) -> list[Image.Image]:
    paths = list(image_paths or [])
    if len(paths) > max_images:
        raise ValueError(f"General Emu3.5 mode accepts at most {max_images} input images.")
    return [resize_pad_image(load_image_rgb(path), image_size) for path in paths]


def run_general_inference(
    config: AppConfig,
    adapter: Emu35Adapter,
    prompt: str,
    image_paths: Iterable[str | Path] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    images = load_general_images(image_paths, config.model.image_size)
    result = adapter.generate_multimodal(prompt, images, config.generation)
    return save_general_result(
        result,
        output_dir,
        prompt=prompt,
        input_image_count=len(images),
        latency_seconds=time.perf_counter() - start,
        save_debug_events=bool(getattr(config.generation, "save_debug_events", True)),
    )


def generated_images_from_result(result: dict[str, Any]) -> list[Image.Image]:
    generated_images = list(result.get("images") or [])
    if not generated_images and result.get("image") is not None:
        generated_images = [result["image"]]
    return generated_images


def save_general_result(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    prompt: str | None = None,
    input_image_count: int | None = None,
    latency_seconds: float | None = None,
    save_debug_events: bool = True,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_text = str(result.get("raw_text", ""))
    (out / "raw_text.txt").write_text(raw_text, encoding="utf-8")
    if prompt is not None:
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")

    generated_images = generated_images_from_result(result)
    image_path: str | None = None
    generated_image_paths: list[str] = []
    if len(generated_images) == 1:
        target = out / "generated_image.png"
        save_image(generated_images[0], target)
        image_path = str(target)
        generated_image_paths.append(str(target))
    elif len(generated_images) > 1:
        generated_dir = out / "generated_images"
        for index, image in enumerate(generated_images):
            target = generated_dir / f"image_{index:03d}.png"
            save_image(image, target)
            generated_image_paths.append(str(target))
        image_path = generated_image_paths[0]

    debug_payload = result.get("debug") or {
        "events": result.get("emu35_events_debug")
        or result.get("metadata", {}).get("event_summaries")
        or [],
    }
    debug_events_path: str | None = None
    if save_debug_events or not generated_images:
        debug_path = out / "emu35_events_debug.json"
        debug_path.write_text(json.dumps(debug_payload, indent=2, default=str), encoding="utf-8")
        debug_events_path = str(debug_path)

    response = {
        "raw_text": raw_text,
        "raw_text_missing": not bool(raw_text),
        "image_path": image_path,
        "generated_image_paths": generated_image_paths,
        "num_generated_images": len(generated_images),
        "image_missing": not bool(generated_images),
        "metadata": result.get("metadata", {}),
        "debug_events_path": debug_events_path,
        "latency_seconds": latency_seconds,
        "input_image_count": input_image_count,
    }
    (out / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    return {**response, "image": generated_images[0] if generated_images else None}


def write_general_error_response(
    output_dir: str | Path,
    diagnostics: dict[str, Any],
    *,
    prompt: str | None = None,
    latency_seconds: float | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if prompt is not None:
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    response = {
        "raw_text": "",
        "raw_text_missing": True,
        "image_path": None,
        "generated_image_paths": [],
        "num_generated_images": 0,
        "image_missing": True,
        "metadata": {},
        "debug_events_path": None,
        "latency_seconds": latency_seconds,
        "error": diagnostics,
    }
    (out / "raw_text.txt").write_text("", encoding="utf-8")
    (out / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    return response


def is_cuda_oom_error(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = f"{type(exc).__name__}: {exc}".lower()
    return "outofmemoryerror" in text or ("cuda" in text and "out of memory" in text)


def cuda_oom_diagnostics(exc: BaseException, *, num_images: int, max_new_tokens: int | None) -> dict[str, Any]:
    return {
        "type": "cuda_oom",
        "message": str(exc),
        "num_input_images": num_images,
        "max_new_tokens": max_new_tokens,
        "suggestion": "Reduce the number of images, image size, or max_new_tokens and retry.",
    }


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
