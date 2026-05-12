from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import typer
from PIL import Image

from src.cad_seq_gen.models.multihead_unet import StructuredMultiHeadUNet
from src.cad_seq_gen.models.step_count import StepCountPredictor
from src.cad_seq_gen.utils.image_ops import save_step_images
from src.cad_seq_gen.utils.runtime_paths import auto_run_dir, discover_latest_checkpoint

app = typer.Typer(add_completion=False)
KEYS = ("prev_depth_map", "sketch_plane_mask", "reference_mask", "result_frame")


def _load_checkpoint(path: Path, device: torch.device) -> tuple[StructuredMultiHeadUNet, Dict]:
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    model = StructuredMultiHeadUNet(
        in_channels=config.get("in_channels", 6),
        base_channels=config.get("base_channels", 32),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, config


def _gray_img_keep_aspect(path: Path, size: int) -> tuple[np.ndarray, dict]:
    img = Image.open(path).convert("L")
    src_w, src_h = img.size
    scale = min(size / max(src_w, 1), size / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = np.zeros((size, size), dtype=np.float32)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = np.array(resized).astype(np.float32) / 255.0
    meta = {"x0": x0, "y0": y0, "new_w": new_w, "new_h": new_h, "src_w": src_w, "src_h": src_h}
    return canvas, meta


def _unpad_to_original_ratio(arr: np.ndarray, meta: dict) -> np.ndarray:
    y0, x0 = meta["y0"], meta["x0"]
    new_h, new_w = meta["new_h"], meta["new_w"]
    src_h, src_w = meta["src_h"], meta["src_w"]
    crop = arr[y0 : y0 + new_h, x0 : x0 + new_w]
    if crop.size == 0:
        crop = arr
    out = Image.fromarray((crop.clip(0.0, 1.0) * 255).astype(np.uint8), mode="L")
    out = out.resize((src_w, src_h), Image.BILINEAR)
    return np.array(out).astype(np.float32) / 255.0


def _to_pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray((x.clip(0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")


@app.command()
def main(
    input_image: Path = typer.Option(..., help="User input CAD part image."),
    checkpoint: Path | None = typer.Option(None, help="best.pt path (auto if omitted)."),
    raw_root: Path = typer.Option(..., help="Raw dataset root for auto step count."),
    processed_root: Path | None = typer.Option(
        None, help="Processed root for auto step count (optional compatibility mode)."
    ),
    output_dir: Path | None = typer.Option(None, help="Output sequence directory (auto if omitted)."),
    num_steps: int = typer.Option(0, help="If 0 then auto-predict."),
    threshold: float = typer.Option(0.5, help="Binary threshold for mask/frame heads."),
    seed: int = typer.Option(123),
    device: str = typer.Option("cuda"),
) -> None:
    _ = seed
    if checkpoint is None:
        checkpoint = discover_latest_checkpoint(raw_root=raw_root)
        if checkpoint is None:
            raise FileNotFoundError("Checkpoint not found. Run training first or pass --checkpoint.")
    if output_dir is None:
        output_dir = auto_run_dir(raw_root=raw_root, mode="infer")

    use_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, cfg = _load_checkpoint(checkpoint, use_device)
    image_size = int(cfg.get("image_size", 384))

    target_part, ratio_meta = _gray_img_keep_aspect(input_image, image_size)
    edges = cv2.Canny((target_part * 255).astype(np.uint8), 80, 180).astype(np.float32) / 255.0

    if num_steps <= 0:
        predictor = StepCountPredictor(device=str(use_device))
        if processed_root is not None:
            predictor.fit(processed_root=processed_root)
        else:
            predictor.fit_from_raw(raw_root=raw_root)
        pred_steps = predictor.predict(_to_pil(target_part), min_steps=1, max_steps=40)
        num_steps = pred_steps

    output_dir.mkdir(parents=True, exist_ok=True)
    prev_state = {k: np.zeros((image_size, image_size), dtype=np.float32) for k in KEYS}

    for idx in range(1, num_steps + 1):
        x = np.stack(
            [
                target_part,
                prev_state["prev_depth_map"],
                prev_state["sketch_plane_mask"],
                prev_state["reference_mask"],
                prev_state["result_frame"],
                edges,
            ],
            axis=0,
        )
        xt = torch.from_numpy(x).unsqueeze(0).to(use_device)
        with torch.no_grad():
            logits = model(xt)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        pred = {
            "prev_depth_map": _unpad_to_original_ratio(probs[0], ratio_meta),
            "sketch_plane_mask": _unpad_to_original_ratio(
                (probs[1] >= threshold).astype(np.float32), ratio_meta
            ),
            "reference_mask": _unpad_to_original_ratio(
                (probs[2] >= threshold).astype(np.float32), ratio_meta
            ),
            "result_frame": _unpad_to_original_ratio(
                (probs[3] >= threshold).astype(np.float32), ratio_meta
            ),
        }
        step_images = {k: _to_pil(v) for k, v in pred.items()}
        save_step_images(step_images, output_dir / f"step_{idx:03d}")
        # Keep autoregressive state in model canvas space.
        prev_state = {
            "prev_depth_map": probs[0],
            "sketch_plane_mask": (probs[1] >= threshold).astype(np.float32),
            "reference_mask": (probs[2] >= threshold).astype(np.float32),
            "result_frame": (probs[3] >= threshold).astype(np.float32),
        }

    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "num_steps": num_steps,
                "image_size": image_size,
                "threshold": threshold,
                "checkpoint": str(checkpoint.as_posix()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Output directory: {output_dir}")
    typer.echo(f"Generated {num_steps} steps to: {output_dir}")


if __name__ == "__main__":
    app()

