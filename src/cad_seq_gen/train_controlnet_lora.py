from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.cad_seq_gen.data.dataset import ControlNetStepDataset

app = typer.Typer(add_completion=False)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.command()
def main(
    processed_root: Path = typer.Option(..., help="Directory containing manifest.jsonl."),
    pretrained_model: str = typer.Option(..., help="Base SD model, e.g. sd-v1.5."),
    controlnet_model: str = typer.Option(..., help="Initial ControlNet model."),
    output_dir: Path = typer.Option(..., help="Output directory."),
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(2),
    lr: float = typer.Option(1e-4),
    num_workers: int = typer.Option(4),
    grad_accum_steps: int = typer.Option(1),
    seed: int = typer.Option(42),
    max_grad_norm: float = typer.Option(1.0),
    mixed_precision: str = typer.Option("fp16"),
) -> None:
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (mixed_precision == "fp16" and device.type == "cuda") else torch.float32

    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = output_dir / "unet_lora"
    controlnet_dir = output_dir / "controlnet"

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(device)
    controlnet = ControlNetModel.from_pretrained(controlnet_model).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()
    unet.to(device, dtype=dtype)

    optimizer = AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)

    train_dataset = ControlNetStepDataset(processed_root=processed_root, split="train")
    if len(train_dataset) == 0:
        raise RuntimeError("Empty train split. Please run prepare_dataset first.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    text_tokens = tokenizer(
        ["cad modeling step canvas"] * batch_size,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    total_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device.type == "cuda"))

    progress = tqdm(total=total_steps, desc="training")
    for _epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            control_values = batch["control_values"].to(device=device, dtype=dtype)

            token_ids = text_tokens["input_ids"][: pixel_values.shape[0]].to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(token_ids)[0]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_values,
                    return_dict=False,
                )

            with torch.cuda.amp.autocast(enabled=(dtype == torch.float16 and device.type == "cuda")):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item() * grad_accum_steps:.5f}")

    unet.save_pretrained(lora_dir)
    controlnet.save_pretrained(controlnet_dir)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model, controlnet=controlnet, torch_dtype=dtype
    )
    pipe.unet = unet.base_model.model
    pipe.save_pretrained(output_dir / "full_pipeline_preview")
    typer.echo(f"Saved LoRA to: {lora_dir}")
    typer.echo(f"Saved ControlNet to: {controlnet_dir}")


if __name__ == "__main__":
    app()

