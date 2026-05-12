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


def _gray_img(path: Path, size: int) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L").resize((size, size), Image.BILINEAR))
    return arr.astype(np.float32) / 255.0


def _to_pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray((x.clip(0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")


@app.command()
def main(
    input_image: Path = typer.Option(..., help="User input CAD part image."),
    checkpoint: Path = typer.Option(..., help="best.pt path."),
    raw_root: Path | None = typer.Option(None, help="Raw dataset root for auto step count."),
    processed_root: Path | None = typer.Option(
        None, help="Processed root for auto step count (optional compatibility mode)."
    ),
    output_dir: Path = typer.Option(..., help="Output sequence directory."),
    num_steps: int = typer.Option(0, help="If 0 then auto-predict."),
    threshold: float = typer.Option(0.5, help="Binary threshold for mask/frame heads."),
    seed: int = typer.Option(123),
    device: str = typer.Option("cuda"),
) -> None:
    _ = seed
    use_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, cfg = _load_checkpoint(checkpoint, use_device)
    image_size = int(cfg.get("image_size", 384))

    target_part = _gray_img(input_image, image_size)
    edges = cv2.Canny((target_part * 255).astype(np.uint8), 80, 180).astype(np.float32) / 255.0

    if num_steps <= 0:
        predictor = StepCountPredictor(device=str(use_device))
        if raw_root is not None:
            predictor.fit_from_raw(raw_root=raw_root)
        elif processed_root is not None:
            predictor.fit(processed_root=processed_root)
        else:
            raise ValueError("num_steps=0 requires either raw_root or processed_root.")
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
            "prev_depth_map": probs[0],
            "sketch_plane_mask": (probs[1] >= threshold).astype(np.float32),
            "reference_mask": (probs[2] >= threshold).astype(np.float32),
            "result_frame": (probs[3] >= threshold).astype(np.float32),
        }
        step_images = {k: _to_pil(v) for k, v in pred.items()}
        save_step_images(step_images, output_dir / f"step_{idx:03d}")
        prev_state = pred

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
    typer.echo(f"Generated {num_steps} steps to: {output_dir}")


if __name__ == "__main__":
    app()

