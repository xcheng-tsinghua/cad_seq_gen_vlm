from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from vision_cad_emu35.config import AppConfig
from vision_cad_emu35.data.collate import CADCollator
from vision_cad_emu35.data.dataset import CADStepDataset
from vision_cad_emu35.models.checkpointing import copy_best_checkpoint, rotate_checkpoints
from vision_cad_emu35.train.losses import scalar_loss_value
from vision_cad_emu35.train.validation import save_validation_samples, validate_loss
from vision_cad_emu35.utils.logging import get_logger
from vision_cad_emu35.utils.seed import seed_everything


LOGGER = get_logger(__name__)


class Emu35Trainer:
    def __init__(self, config: AppConfig, adapter: Any) -> None:
        self.config = config
        self.adapter = adapter
        self.output_dir = Path(config.data.output_dir)
        self.global_step = 0
        self.best_val_loss = math.inf

    def train(self) -> None:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise ImportError("torch is required for training.") from exc

        seed_everything(self.config.training.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "run_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2),
            encoding="utf-8",
        )

        self.adapter.load_model()
        if self.config.training.resume_from_checkpoint:
            self.adapter.load_checkpoint(self.config.training.resume_from_checkpoint)
        if self.config.training.compile_model and hasattr(torch, "compile"):
            self.adapter.model = torch.compile(self.adapter.model)

        train_manifest = Path(self.config.data.manifest_dir) / "train.jsonl"
        val_manifest = Path(self.config.data.manifest_dir) / "val.jsonl"
        train_dataset = CADStepDataset(train_manifest, image_size=self.config.data.image_size)
        val_dataset = CADStepDataset(val_manifest, image_size=self.config.data.image_size) if val_manifest.exists() else None
        collate = CADCollator(self.adapter)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size_per_device,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size_per_device,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                collate_fn=collate,
                pin_memory=torch.cuda.is_available(),
            )
            if val_dataset is not None
            else None
        )

        optimizer = torch.optim.AdamW(
            [p for p in self.adapter.model.parameters() if p.requires_grad],
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        total_updates = math.ceil(len(train_loader) * self.config.training.epochs / self.config.training.gradient_accumulation_steps)
        scheduler = self._build_scheduler(optimizer, total_updates)
        writer = self._build_tensorboard_writer()
        wandb_run = self._build_wandb_run()
        scaler = self._build_grad_scaler(torch)

        self.adapter.model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = self.config.training.gradient_accumulation_steps

        for epoch in range(self.config.training.epochs):
            for batch_idx, batch in enumerate(train_loader):
                with self._autocast_context(torch):
                    loss = self.adapter.forward_loss(batch) / accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % accum != 0:
                    continue

                if scaler is not None:
                    scaler.unscale_(optimizer)
                if self.config.training.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.adapter.model.parameters(), self.config.training.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                loss_value = scalar_loss_value(loss) * accum
                if self.global_step % self.config.training.logging_steps == 0:
                    LOGGER.info("epoch=%s step=%s train_loss=%.6f", epoch, self.global_step, loss_value)
                    if writer:
                        writer.add_scalar("train/loss", loss_value, self.global_step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], self.global_step)
                    if wandb_run:
                        wandb_run.log({"train/loss": loss_value, "train/lr": scheduler.get_last_lr()[0], "step": self.global_step})

                if val_loader is not None and self.global_step % self.config.training.validation_steps == 0:
                    val_loss = validate_loss(self.adapter, val_loader)
                    LOGGER.info("step=%s val_loss=%.6f", self.global_step, val_loss)
                    if writer:
                        writer.add_scalar("val/loss", val_loss, self.global_step)
                    if wandb_run:
                        wandb_run.log({"val/loss": val_loss, "step": self.global_step})
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        best_source = self._save_checkpoint(f"checkpoint-{self.global_step}")
                        copy_best_checkpoint(best_source, self.output_dir)
                    save_validation_samples(
                        self.adapter,
                        val_dataset,
                        self.output_dir / "qualitative" / f"step_{self.global_step}",
                        self.config.generation,
                    )

                if self.global_step % self.config.training.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")
                    self._save_checkpoint("latest")
                    rotate_checkpoints(self.output_dir, self.config.training.save_total_limit)

        self._save_checkpoint("latest")
        if writer:
            writer.close()
        if wandb_run:
            wandb_run.finish()

    def _save_checkpoint(self, name: str) -> Path:
        checkpoint_dir = self.output_dir / name
        self.adapter.save_checkpoint(checkpoint_dir)
        return checkpoint_dir

    def _build_scheduler(self, optimizer: Any, total_updates: int) -> Any:
        import torch

        warmup_steps = int(total_updates * self.config.training.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_updates - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _build_tensorboard_writer(self) -> Any | None:
        if not self.config.logging.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            LOGGER.warning("TensorBoard is enabled but tensorboard is not installed.")
            return None
        return SummaryWriter(log_dir=str(self.output_dir / "tb"))

    def _build_wandb_run(self) -> Any | None:
        if not self.config.logging.wandb:
            return None
        try:
            import wandb
        except Exception:
            LOGGER.warning("Weights & Biases is enabled but wandb is not installed.")
            return None
        return wandb.init(project=self.config.logging.project_name, config=self.config.to_dict())

    def _build_grad_scaler(self, torch: Any) -> Any | None:
        if self.config.training.mixed_precision != "fp16" or not torch.cuda.is_available():
            return None
        return torch.cuda.amp.GradScaler()

    def _autocast_context(self, torch: Any) -> Any:
        from contextlib import nullcontext

        if not torch.cuda.is_available():
            return nullcontext()
        precision = self.config.training.mixed_precision
        if precision == "bf16":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if precision == "fp16":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

