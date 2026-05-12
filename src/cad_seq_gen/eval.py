from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from src.cad_seq_gen.data.structured_dataset import StructuredStepDataset
from src.cad_seq_gen.models.multihead_unet import StructuredMultiHeadUNet

app = typer.Typer(add_completion=False)
HEADS = ("prev_depth_map", "sketch_plane_mask", "reference_mask", "result_frame")


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[StructuredMultiHeadUNet, Dict]:
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


def _bin_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, np.logical_not(gt_b)).sum())
    fn = float(np.logical_and(np.logical_not(pred_b), gt_b).sum())
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def _depth_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    mae = float(np.mean(np.abs(pred - gt)))
    mse = float(np.mean((pred - gt) ** 2))
    psnr = float(10.0 * np.log10(1.0 / (mse + eps)))
    return {"mae": mae, "psnr": psnr}


def _wire_edge_f1(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    pred_edge = cv2.Canny((pred_bin * 255).astype(np.uint8), 50, 150) > 0
    gt_edge = cv2.Canny((gt_bin * 255).astype(np.uint8), 50, 150) > 0
    m = _bin_metrics(pred_edge.astype(np.uint8), gt_edge.astype(np.uint8))
    return m["f1"]


def _save_visual(
    out_path: Path,
    inputs: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    sample_id: str,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    fig.suptitle(sample_id)
    rows = list(range(4))
    for i in rows:
        axes[i, 0].imshow(inputs[0], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"target_part ({HEADS[i]})")
        axes[i, 1].imshow(pred[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("pred")
        axes[i, 2].imshow(gt[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("gt")
        for j in range(3):
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@app.command()
def main(
    raw_root: Path | None = typer.Option(None, help="Raw dataset root (recommended)."),
    processed_root: Path | None = typer.Option(
        None, help="Processed root with manifest.jsonl (optional compatibility mode)."
    ),
    checkpoint: Path = typer.Option(..., help="best.pt path."),
    output_dir: Path = typer.Option(..., help="Evaluation output directory."),
    image_size: int = typer.Option(384),
    batch_size: int = typer.Option(8),
    num_workers: int = typer.Option(4),
    val_ratio: float = typer.Option(0.1, help="Used only when loading from raw_root."),
    threshold: float = typer.Option(0.5),
    max_visuals: int = typer.Option(40),
    seed: int = typer.Option(42),
    device: str = typer.Option("cuda"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visuals"
    vis_dir.mkdir(exist_ok=True)

    use_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, _ = _load_checkpoint(checkpoint, use_device)
    val_ds = StructuredStepDataset(
        processed_root=processed_root,
        raw_root=raw_root,
        split="val",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(use_device.type == "cuda"),
    )

    agg = {
        "prev_depth_map": {"mae": 0.0, "psnr": 0.0},
        "sketch_plane_mask": {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        "reference_mask": {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        "result_frame": {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "edge_f1": 0.0},
    }
    n = 0
    vis_saved = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["inputs"].to(use_device)
            y = batch["targets"].to(use_device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            y_np = y.cpu().numpy()
            x_np = x.cpu().numpy()

            for i in range(probs.shape[0]):
                pred = probs[i]
                gt = y_np[i]
                sample_id = batch["sample_id"][i]

                pred_depth = pred[0]
                gt_depth = gt[0]
                dmap = _depth_metrics(pred_depth, gt_depth)
                agg["prev_depth_map"]["mae"] += dmap["mae"]
                agg["prev_depth_map"]["psnr"] += dmap["psnr"]

                pred_sketch = (pred[1] >= threshold).astype(np.uint8)
                gt_sketch = (gt[1] >= threshold).astype(np.uint8)
                smap = _bin_metrics(pred_sketch, gt_sketch)
                for k, v in smap.items():
                    agg["sketch_plane_mask"][k] += v

                pred_ref = (pred[2] >= threshold).astype(np.uint8)
                gt_ref = (gt[2] >= threshold).astype(np.uint8)
                rmap = _bin_metrics(pred_ref, gt_ref)
                for k, v in rmap.items():
                    agg["reference_mask"][k] += v

                pred_wire = (pred[3] >= threshold).astype(np.uint8)
                gt_wire = (gt[3] >= threshold).astype(np.uint8)
                wmap = _bin_metrics(pred_wire, gt_wire)
                for k, v in wmap.items():
                    agg["result_frame"][k] += v
                agg["result_frame"]["edge_f1"] += _wire_edge_f1(pred_wire, gt_wire)

                if vis_saved < max_visuals:
                    _save_visual(vis_dir / f"{sample_id}.png", x_np[i], pred, gt, sample_id)
                    vis_saved += 1
                n += 1

    if n == 0:
        raise RuntimeError("Validation split is empty.")

    metrics = {}
    for head, m in agg.items():
        metrics[head] = {k: float(v / n) for k, v in m.items()}

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    labels = ["sketch_iou", "reference_iou", "wire_iou", "wire_edge_f1"]
    values = [
        metrics["sketch_plane_mask"]["iou"],
        metrics["reference_mask"]["iou"],
        metrics["result_frame"]["iou"],
        metrics["result_frame"]["edge_f1"],
    ]
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_title("Validation Metrics")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_metrics.png", dpi=150)
    plt.close(fig)

    typer.echo(f"Done. Evaluated {n} samples. metrics.json saved to {output_dir}")


if __name__ == "__main__":
    app()

