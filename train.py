from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import typer
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.structured_dataset import StructuredStepDataset
from models.losses import StructuredLoss, sd_latent_consistency_loss
from models.multihead_unet import StructuredMultiHeadUNet
from utils.runtime_paths import auto_run_dir, project_root, save_latest_checkpoint

app = typer.Typer(add_completion=False)
DEFAULT_SD_MODEL = "stable-diffusion-3.5-medium"
DEFAULT_SD_REPO = "stabilityai/stable-diffusion-3.5-medium"
TOKEN_FILE = project_root() / "vlm" / "hf_access_token.json"
SD35_DIR = project_root() / "vlm" / DEFAULT_SD_MODEL
SD35_WEIGHT_PATH = SD35_DIR / "sd3.5_medium.safetensors"
SD35_CONFIG_PATH = SD35_DIR / "sd3.5_medium_config.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_hf_token_from_file(token_file: Path = TOKEN_FILE) -> str | None:
    """Read Hugging Face access token from JSON file under vlm/."""
    if not token_file.exists():
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(
            json.dumps(
                {
                    "access_token": "",
                    "note": "Put your Hugging Face token here, then rerun training.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        typer.echo(f"[warn] Token file created: {token_file}. Please fill access_token.")
        return None
    try:
        payload = json.loads(token_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON token file: {token_file}. {exc}") from exc
    token = str(payload.get("access_token", "")).strip()
    return token or None


def _hf_login_if_needed(token: str | None) -> None:
    """Prepare HuggingFace auth environment using token from JSON file."""
    if not token:
        typer.echo(
            "[warn] No Hugging Face token provided. Gated/private repos "
            "(e.g. SD 3.5 medium) may fail to download. "
            f"Please set access_token in: {TOKEN_FILE}"
        )
        return
    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    typer.echo("[info] Hugging Face token loaded from JSON file.")


def _repo_id_hint(model_id_or_path: str) -> str | None:
    # Full HF repo id
    if "/" in model_id_or_path:
        return model_id_or_path
    # Local short name convention in this project
    if model_id_or_path == DEFAULT_SD_MODEL:
        return DEFAULT_SD_REPO
    return None


def _is_vae_compatible_weight(weight_path: Path) -> bool:
    """Heuristically verify whether a safetensors file is VAE-only weights."""
    try:
        with safe_open(str(weight_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        return False
    if not keys:
        return False
    vae_prefixes = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")
    return any(k.startswith(vae_prefixes) for k in keys)


def _ensure_sd35_assets(model_id_or_path: str, token: str | None) -> tuple[Path, Path]:
    """Ensure SD3.5 weight+config exist at fixed paths under vlm/."""
    SD35_DIR.mkdir(parents=True, exist_ok=True)
    repo_id = _repo_id_hint(model_id_or_path) or model_id_or_path
    need_download_weight = (not SD35_WEIGHT_PATH.exists()) or (not _is_vae_compatible_weight(SD35_WEIGHT_PATH))
    if need_download_weight:
        if SD35_WEIGHT_PATH.exists():
            typer.echo(
                f"[warn] Existing weight at {SD35_WEIGHT_PATH} is not VAE-compatible. "
                "Will overwrite with HF vae/diffusion_pytorch_model.safetensors."
            )
        downloaded_weight = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="vae/diffusion_pytorch_model.safetensors",
                token=token,
            )
        )
        shutil.copy2(downloaded_weight, SD35_WEIGHT_PATH)
        typer.echo(f"[info] downloaded SD3.5 VAE weights to: {SD35_WEIGHT_PATH}")

    if not SD35_CONFIG_PATH.exists():
        # Use VAE config and persist to the required fixed filename.
        downloaded_cfg = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="vae/config.json",
                token=token,
            )
        )
        shutil.copy2(downloaded_cfg, SD35_CONFIG_PATH)
        typer.echo(f"[info] downloaded SD3.5 config to: {SD35_CONFIG_PATH}")
    return SD35_WEIGHT_PATH, SD35_CONFIG_PATH


def _extract_vae_state_dict_from_sd35_single_file(weight_path: Path) -> Dict[str, torch.Tensor]:
    """Load VAE state_dict from fixed local safetensors file."""
    return load_safetensors(str(weight_path), device="cpu")


def _load_sd_vae(model_id: str, device: torch.device, token: str | None) -> Any:
    """Create VAE from the fixed SD3.5 config+weight paths under vlm/."""
    weight_path, config_path = _ensure_sd35_assets(model_id_or_path=model_id, token=token)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    vae = AutoencoderKL.from_config(cfg)
    state_dict = _extract_vae_state_dict_from_sd35_single_file(weight_path)

    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    expected = len(vae.state_dict())
    if expected > 0 and len(missing) > 0.5 * expected:
        raise RuntimeError(
            f"VAE state_dict mismatch: missing={len(missing)}/{expected}. "
            f"Weight file is incompatible: {weight_path}. "
            "Please ensure token is valid and rerun to redownload VAE weights."
        )
    if missing or unexpected:
        typer.echo(
            f"[warn] VAE state_dict partial match: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
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
    vae: Any | None = None,
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
    raw_root: Path = typer.Option('/opt/data/private/data_set/cad_seq_img', help="Raw dataset root."),
    processed_root: Path | None = typer.Option(None, help="Processed root with manifest.jsonl (optional compatibility mode)."),
    output_dir: Path | None = typer.Option(None, help="Training output directory (auto if omitted)."),
    image_size: int = typer.Option(384),
    base_channels: int = typer.Option(32),
    epochs: int = typer.Option(80),
    batch_size: int = typer.Option(8),
    lr: float = typer.Option(2e-4),
    weight_decay: float = typer.Option(1e-4),
    num_workers: int = typer.Option(4),
    seed: int = typer.Option(42),
    val_ratio: float = typer.Option(0.1, help="Used only when loading from raw_root."),
    device: str = typer.Option("cuda"),
    sd_model_id: str = typer.Option(DEFAULT_SD_MODEL, help="HF repo id (default stabilityai/stable-diffusion-3.5-medium)."),
    w_sd_latent: float = typer.Option(0.2, help="Weight of SD latent consistency loss."),
) -> None:
    seed_everything(seed)
    token = _load_hf_token_from_file()
    if w_sd_latent > 0:
        _hf_login_if_needed(token)
    if output_dir is None:
        output_dir = auto_run_dir(raw_root=raw_root, mode="train")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = StructuredMultiHeadUNet(in_channels=6, base_channels=base_channels).to(use_device)
    loss_fn = StructuredLoss().to(use_device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    vae = _load_sd_vae(sd_model_id, use_device, token) if w_sd_latent > 0 else None

    train_ds = StructuredStepDataset(
        processed_root=processed_root,
        raw_root=raw_root,
        split="train",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    val_ds = StructuredStepDataset(
        processed_root=processed_root,
        raw_root=raw_root,
        split="val",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )
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
                output_dir / "best.pth",
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
            output_dir / "last.pth",
        )

        typer.echo(
            f"epoch={epoch} train_total={train_log['train_total']:.4f} val_total={val_log['val_total']:.4f}"
        )

    (output_dir / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_ckpt = output_dir / "best.pth"
    if best_ckpt.exists():
        save_latest_checkpoint(raw_root=raw_root, checkpoint=best_ckpt)
        typer.echo(f"Latest checkpoint marker updated: {best_ckpt}")
    typer.echo(f"Output directory: {output_dir}")
    typer.echo(f"Done. best_val_total={best_val:.4f}")


if __name__ == "__main__":
    app()
