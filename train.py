"""Lightweight instruction-tuning trainer for the multi-view CAD generator.

Frozen parts of the network:
    * VAE
    * Both SDXL text encoders
    * The CLIP-Vision image encoder
    * UNet base weights (only LoRA deltas train)

Trainable parts (verified by inspecting ``requires_grad``):
    1. LoRA weights injected into the UNet (via PEFT).
    2. Every parameter of :class:`MultiViewControlNetModel`.
    3. :class:`ImageProjModel` + the IP-Adapter ``to_k_ip / to_v_ip`` linears.

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
from dataset import CADMultiViewDataset, collate_cad_batch, make_worker_init_fn
from models import CADMultiViewPipeline


# ---------------------------------------------------------------------------
def _count_trainable(named_params) -> int:
    return sum(p.numel() for n, p in named_params if p.requires_grad)


def _save_checkpoint(
    pipeline: CADMultiViewPipeline,
    accelerator: Accelerator,
    ckpt_dir: str,
) -> None:
    """Unwrap accelerate/DDP wrappers and call :meth:`save_trainables`.

    Without unwrapping, ``named_parameters()`` keys would carry a ``module.``
    prefix under DDP -- :meth:`load_trainables` would then silently miss them.
    """
    pipeline.save_trainables(
        ckpt_dir,
        unet_module=accelerator.unwrap_model(pipeline.unet),
        mv_controlnet_module=accelerator.unwrap_model(pipeline.mv_controlnet),
        image_proj_module=accelerator.unwrap_model(pipeline.image_proj_model),
    )


def _log_trainable_breakdown(pipeline: CADMultiViewPipeline, logger: logging.Logger) -> None:
    """Print a sanity check of which sub-modules are trainable and at what size."""
    lora_params, mvcn_params, ipproj_params, ipattn_params, frozen = 0, 0, 0, 0, 0

    for n, p in pipeline.unet.named_parameters():
        if not p.requires_grad:
            frozen += p.numel()
            continue
        if "lora_" in n:
            lora_params += p.numel()
        elif "to_k_ip" in n or "to_v_ip" in n:
            ipattn_params += p.numel()
        else:
            # Anything left over that's trainable in the UNet would be a bug
            # in our freeze logic.
            logger.warning("Unexpected trainable UNet param: %s (%d)", n, p.numel())
            ipattn_params += p.numel()

    for p in pipeline.mv_controlnet.parameters():
        if p.requires_grad:
            mvcn_params += p.numel()
        else:
            frozen += p.numel()

    for p in pipeline.image_proj_model.parameters():
        if p.requires_grad:
            ipproj_params += p.numel()

    for m in (pipeline.vae, pipeline.text_encoder_one, pipeline.text_encoder_two,
              pipeline.image_encoder):
        frozen += sum(p.numel() for p in m.parameters())

    logger.info("=" * 60)
    logger.info("Trainable parameter breakdown:")
    logger.info("  LoRA (UNet):              %.2fM", lora_params / 1e6)
    logger.info("  IP-Adapter K/V (UNet):    %.2fM", ipattn_params / 1e6)
    logger.info("  Image projection model:   %.2fM", ipproj_params / 1e6)
    logger.info("  Multi-View ControlNet:    %.2fM", mvcn_params / 1e6)
    logger.info("  Frozen (VAE+TextEnc+CLIP+UNet base): %.2fM", frozen / 1e6)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
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

    # ---------------- pipeline ------------------------------------------------
    weight_dtype = {
        "no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    }[train_cfg.mixed_precision]
    logger.info("Instantiating pipeline (this downloads SDXL on first run)...")
    pipeline = CADMultiViewPipeline(
        model_cfg=model_cfg,
        device=accelerator.device,
        weight_dtype=weight_dtype,
    )
    pipeline.to_device(accelerator.device)
    _log_trainable_breakdown(pipeline, logger)

    # ---------------- dataset -------------------------------------------------
    train_ds = CADMultiViewDataset(
        data_root=data_cfg.data_root,
        random_i_final_view=data_cfg.random_i_final_view,
        require_all_views=data_cfg.require_all_views,
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
        # Spread the seed across workers so each picks distinct random views.
        worker_init_fn=make_worker_init_fn(base_seed=train_cfg.seed),
    )

    # ---------------- optimizer ----------------------------------------------
    params = pipeline.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    total_steps = math.ceil(
        len(train_dl) / train_cfg.gradient_accumulation_steps
    ) * train_cfg.num_train_epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, s / max(1, train_cfg.lr_warmup_steps)),
    )

    # Only register the *trainable* sub-modules with accelerate. The frozen
    # ones already live on `accelerator.device` after `to_device()`.
    pipeline.unet, pipeline.mv_controlnet, pipeline.image_proj_model, \
        optimizer, train_dl, lr_scheduler = accelerator.prepare(
            pipeline.unet,
            pipeline.mv_controlnet,
            pipeline.image_proj_model,
            optimizer,
            train_dl,
            lr_scheduler,
        )

    # ---------------- training loop ------------------------------------------
    global_step = 0
    progress = tqdm(
        range(total_steps),
        disable=not accelerator.is_local_main_process,
        desc="train",
    )

    for epoch in range(train_cfg.num_train_epochs):
        for batch in train_dl:
            with accelerator.accumulate(pipeline.unet):
                i_final  = batch["I_final"].to(accelerator.device,  non_blocking=True)
                g_prev   = batch["G_prev"].to(accelerator.device,   non_blocking=True)
                g_target = batch["G_target"].to(accelerator.device, non_blocking=True)
                prompts: List[str] = batch["prompt"]

                loss = pipeline.training_step_loss(
                    i_final=i_final,
                    g_prev=g_prev,
                    g_target=g_target,
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

                if (
                    global_step % train_cfg.save_every == 0
                    and accelerator.is_main_process
                ):
                    ckpt_dir = os.path.join(train_cfg.output_dir, f"step_{global_step:07d}")
                    logger.info("Saving trainables to %s", ckpt_dir)
                    _save_checkpoint(pipeline, accelerator, ckpt_dir)

    # Final save.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(train_cfg.output_dir, "final")
        logger.info("Training done. Saving final checkpoint to %s", final_dir)
        _save_checkpoint(pipeline, accelerator, final_dir)


if __name__ == "__main__":
    main()
