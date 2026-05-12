from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import typer
from diffusers import AutoencoderKL
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cad_seq_gen.data.structured_dataset import StructuredStepDataset
from src.cad_seq_gen.models.losses import StructuredLoss, sd_latent_consistency_loss
from src.cad_seq_gen.models.multihead_unet import StructuredMultiHeadUNet

app = typer.Typer(add_completion=False)
DEFAULT_SD_MODEL = "stabilityai/stable-diffusion-3.5-medium"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_sd_vae(model_id: str, device: torch.device) -> AutoencoderKL:
    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    except Exception:
        vae = AutoencoderKL.from_pretrained(model_id)
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def evaluate(
    model: StructuredMultiHeadUNet,
    loader: DataLoader,
    loss_fn: StructuredLoss,
    device: torch.device,
    vae: AutoencoderKL | None = None,
    w_sd_latent: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    agg = {
        "total": 0.0,
        "depth": 0.0,
        "sketch": 0.0,
        "reference": 0.0,
        "wire": 0.0,
        "wire_edge": 0.0,
        "sd_latent": 0.0,
    }
    count = 0
    for batch in loader:
        x = batch["inputs"].to(device)
        y = batch["targets"].to(device)
        logits = model(x)
        loss_map = loss_fn(logits, y)
        total = loss_map["total"]
        sd_lat = torch.tensor(0.0, device=device)
        if vae is not None and w_sd_latent > 0:
            sd_lat = sd_latent_consistency_loss(vae=vae, pred_logits=logits, target=y, head_index=3)
            total = total + w_sd_latent * sd_lat
        bsz = x.shape[0]
        agg["total"] += float(total.item()) * bsz
        agg["sd_latent"] += float(sd_lat.item()) * bsz
        for k in ("depth", "sketch", "reference", "wire", "wire_edge"):
            agg[k] += float(loss_map[k].item()) * bsz
        count += bsz
    if count == 0:
        return {k: 0.0 for k in agg}
    return {k: v / count for k, v in agg.items()}


@app.command()
def main(
    processed_root: Path = typer.Option(..., help="Root that contains manifest.jsonl."),
    output_dir: Path = typer.Option(..., help="Training output directory."),
    image_size: int = typer.Option(384),
    base_channels: int = typer.Option(32),
    epochs: int = typer.Option(80),
    batch_size: int = typer.Option(8),
    lr: float = typer.Option(2e-4),
    weight_decay: float = typer.Option(1e-4),
    num_workers: int = typer.Option(4),
    seed: int = typer.Option(42),
    device: str = typer.Option("cuda"),
    sd_model_id: str = typer.Option(DEFAULT_SD_MODEL, help="Latest Stable Diffusion model id."),
    w_sd_latent: float = typer.Option(0.2, help="Weight of SD latent consistency loss."),
) -> None:
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = StructuredMultiHeadUNet(in_channels=6, base_channels=base_channels).to(use_device)
    loss_fn = StructuredLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    vae = _load_sd_vae(sd_model_id, use_device) if w_sd_latent > 0 else None

    train_ds = StructuredStepDataset(processed_root=processed_root, split="train", image_size=image_size)
    val_ds = StructuredStepDataset(processed_root=processed_root, split="val", image_size=image_size)
    if len(train_ds) == 0:
        raise RuntimeError("Empty training split for structured dataset.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(use_device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(use_device.type == "cuda"),
    )

    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            "total": 0.0,
            "depth": 0.0,
            "sketch": 0.0,
            "reference": 0.0,
            "wire": 0.0,
            "wire_edge": 0.0,
            "sd_latent": 0.0,
        }
        seen = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            x = batch["inputs"].to(use_device)
            y = batch["targets"].to(use_device)
            logits = model(x)
            loss_map = loss_fn(logits, y)
            total_loss = loss_map["total"]

            sd_lat = torch.tensor(0.0, device=use_device)
            if vae is not None and w_sd_latent > 0:
                sd_lat = sd_latent_consistency_loss(vae=vae, pred_logits=logits, target=y, head_index=3)
                total_loss = total_loss + w_sd_latent * sd_lat

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = x.shape[0]
            running["total"] += float(total_loss.item()) * bsz
            running["sd_latent"] += float(sd_lat.item()) * bsz
            for k in ("depth", "sketch", "reference", "wire", "wire_edge"):
                running[k] += float(loss_map[k].item()) * bsz
            seen += bsz
            pbar.set_postfix(loss=f"{(running['total'] / max(seen, 1)):.4f}")

        train_log = {f"train_{k}": (running[k] / max(seen, 1)) for k in running}
        val_log_raw = evaluate(model, val_loader, loss_fn, use_device, vae=vae, w_sd_latent=w_sd_latent)
        val_log = {f"val_{k}": v for k, v in val_log_raw.items()}
        row = {"epoch": epoch, **train_log, **val_log}
        history.append(row)

        if val_log["val_total"] < best_val:
            best_val = val_log["val_total"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "in_channels": 6,
                        "base_channels": base_channels,
                        "image_size": image_size,
                    },
                    "best_val_total": best_val,
                    "epoch": epoch,
                    "sd_model_id": sd_model_id,
                    "w_sd_latent": w_sd_latent,
                },
                output_dir / "best.pt",
            )
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {
                    "in_channels": 6,
                    "base_channels": base_channels,
                    "image_size": image_size,
                },
                "epoch": epoch,
                "sd_model_id": sd_model_id,
                "w_sd_latent": w_sd_latent,
            },
            output_dir / "last.pt",
        )

        typer.echo(
            f"epoch={epoch} train_total={train_log['train_total']:.4f} val_total={val_log['val_total']:.4f}"
        )

    (output_dir / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"Done. best_val_total={best_val:.4f}")


if __name__ == "__main__":
    app()

