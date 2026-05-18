"""Lightweight trainer — Phase 2 (diffusion painter).

Finetunes **SDXL + standard ControlNet + IP-Adapter** on single-view CAD steps.
The trainable stack mirrors ``diffusers.StableDiffusionXLControlNetPipeline`` components
(VAE, dual CLIP text encoders, ``ControlNetModel``, ``UNet2DConditionModel``); this script
uses :class:`models.CADSingleViewPipeline` for a compact training forward (noise MSE) instead
of wrapping the full pipeline class.

**Conditioning:** ``prev_depth_map`` (3-ch) + ``prompt.txt`` + ``final_snapshot`` (IP-Adapter).
**Target:** ``overlayed_all.png`` for the current step.

// Phase 1 labels prompts via ``auto_label.py``; Phase 3 planner SFT is ``train_qwen_planner.py``.

Usage::

    accelerate launch train.py
"""

from __future__ import annotations

import logging
import math
import os
from typing import List

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import DataConfig, ModelConfig, TrainConfig
from dataset import CADSingleViewDataset, collate_cad_batch, make_worker_init_fn
from models import CADSingleViewPipeline


def _save_checkpoint(
    pipeline: CADSingleViewPipeline,
    accelerator: Accelerator,
    ckpt_dir: str,
) -> None:
    pipeline.save_trainables(
        ckpt_dir,
        unet_module=accelerator.unwrap_model(pipeline.unet),
        controlnet_module=accelerator.unwrap_model(pipeline.controlnet),
        image_proj_module=accelerator.unwrap_model(pipeline.image_proj_model),
    )


def _log_trainable_breakdown(pipeline: CADSingleViewPipeline, logger: logging.Logger) -> None:
    lora_params, cn_params, ipproj_params, ipattn_params, frozen = 0, 0, 0, 0, 0

    for n, p in pipeline.unet.named_parameters():
        if not p.requires_grad:
            frozen += p.numel()
            continue
        if "lora_" in n:
            lora_params += p.numel()
        elif "to_k_ip" in n or "to_v_ip" in n:
            ipattn_params += p.numel()
        else:
            logger.warning("Unexpected trainable UNet param: %s (%d)", n, p.numel())
            ipattn_params += p.numel()

    for p in pipeline.controlnet.parameters():
        if p.requires_grad:
            cn_params += p.numel()
        else:
            frozen += p.numel()

    for p in pipeline.image_proj_model.parameters():
        if p.requires_grad:
            ipproj_params += p.numel()

    for m in (
        pipeline.vae,
        pipeline.text_encoder_one,
        pipeline.text_encoder_two,
        pipeline.image_encoder,
    ):
        frozen += sum(p.numel() for p in m.parameters())

    logger.info("=" * 60)
    logger.info("Trainable parameter breakdown (MVP):")
    logger.info("  LoRA (UNet):              %.2fM", lora_params / 1e6)
    logger.info("  IP-Adapter K/V (UNet):    %.2fM", ipattn_params / 1e6)
    logger.info("  Image projection model:   %.2fM", ipproj_params / 1e6)
    logger.info("  ControlNet (standard):    %.2fM", cn_params / 1e6)
    logger.info("  Frozen (VAE+TextEnc+CLIP+UNet base): %.2fM", frozen / 1e6)
    logger.info("=" * 60)


def main() -> None:
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    data_cfg = DataConfig()

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        mixed_precision=train_cfg.mixed_precision,
        log_with=None,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO if accelerator.is_main_process else logging.WARN,
    )
    logger = logging.getLogger("cad_train")
    set_seed(train_cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(train_cfg.output_dir, exist_ok=True)

    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        train_cfg.mixed_precision
    ]
    logger.info("Instantiating single-view pipeline (downloads on first run)...")
    pipeline = CADSingleViewPipeline(
        model_cfg=model_cfg,
        device=accelerator.device,
        weight_dtype=weight_dtype,
    )
    pipeline.to_device(accelerator.device)
    _log_trainable_breakdown(pipeline, logger)

    train_ds = CADSingleViewDataset(
        data_root=data_cfg.data_root,
        part_ids_file=data_cfg.part_ids_file,
        seed=train_cfg.seed,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg.train_batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_cad_batch,
        drop_last=data_cfg.drop_last,
        pin_memory=True,
        worker_init_fn=make_worker_init_fn(base_seed=train_cfg.seed),
    )

    optimizer = torch.optim.AdamW(
        pipeline.get_trainable_parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    total_steps = math.ceil(len(train_dl) / train_cfg.gradient_accumulation_steps) * train_cfg.num_train_epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, s / max(1, train_cfg.lr_warmup_steps)),
    )

    pipeline.unet, pipeline.controlnet, pipeline.image_proj_model, optimizer, train_dl, lr_scheduler = (
        accelerator.prepare(
            pipeline.unet,
            pipeline.controlnet,
            pipeline.image_proj_model,
            optimizer,
            train_dl,
            lr_scheduler,
        )
    )

    global_step = 0
    progress = tqdm(
        range(total_steps),
        disable=not accelerator.is_local_main_process,
        desc="train",
    )

    for epoch in range(train_cfg.num_train_epochs):
        for batch in train_dl:
            with accelerator.accumulate(pipeline.unet):
                i_final = batch["I_final"].to(accelerator.device, non_blocking=True)
                condition = batch["condition_image"].to(accelerator.device, non_blocking=True)
                target = batch["target_image"].to(accelerator.device, non_blocking=True)
                prompts: List[str] = batch["prompt"]

                loss = pipeline.training_step_loss(
                    i_final=i_final,
                    condition_image=condition,
                    target_image=target,
                    prompts=prompts,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        pipeline.get_trainable_parameters(),
                        train_cfg.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % train_cfg.log_every == 0:
                    progress.set_postfix(
                        loss=f"{loss.detach().float().item():.4f}",
                        epoch=epoch,
                        step=global_step,
                    )
                if global_step % train_cfg.save_every == 0 and accelerator.is_main_process:
                    ckpt_dir = os.path.join(train_cfg.output_dir, f"step_{global_step:07d}")
                    logger.info("Saving trainables to %s", ckpt_dir)
                    _save_checkpoint(pipeline, accelerator, ckpt_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(train_cfg.output_dir, "final")
        logger.info("Training done. Saving final checkpoint to %s", final_dir)
        _save_checkpoint(pipeline, accelerator, final_dir)


if __name__ == "__main__":
    main()
