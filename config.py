"""Global configuration — single-view MVP CAD reverse-modeling pipeline.

// MVP Refactor: multi-view (8 cameras, grids, cross-view attention) removed.
Only ``[PART_ID]_PPP/`` is supported; train one canonical view end-to-end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


PRETRAINED_DIR: str = str((Path(__file__).resolve().parent / "pretrained_lm").resolve())


def set_hf_cache_env(force: bool = False) -> str:
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    cache_keys = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
    for k in cache_keys:
        if force or not os.environ.get(k):
            os.environ[k] = PRETRAINED_DIR
    if force or not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = PRETRAINED_DIR
    return PRETRAINED_DIR


set_hf_cache_env()

# ---------------------------------------------------------------------------
# Data layout (single view: PPP only)
# ---------------------------------------------------------------------------
MVP_VIEW_SUFFIX: str = "PPP"  # // MVP Refactor: hardcoded single view

I_FINAL_FILENAME: str = "final_snapshot.png"
STEP_DIR_PREFIX: str = "roll_back_index_"
PROMPT_FILENAME: str = "prompt.txt"
PREV_DEPTH_FILENAME: str = "prev_depth_map.png"
OVERLAYED_FILENAME: str = "overlayed_all.png"
OPERATION_PARAM_FILENAME: str = "operation_param.json"

# Training image resolution (condition + target). SDXL-friendly default.
TRAIN_IMAGE_H: int = 1024
TRAIN_IMAGE_W: int = 1024

IFINAL_H: int = 512
IFINAL_W: int = 512


@dataclass
class ModelConfig:
    """SDXL + standard ControlNet + IP-Adapter (custom projector, diffusers UNet)."""

    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # // MVP Refactor: stock diffusers ControlNet (depth conditioning for prev_depth).
    controlnet_model_name_or_path: str = "diffusers/controlnet-depth-sdxl-1.0"

    clip_image_encoder_name_or_path: str = "openai/clip-vit-large-patch14"

    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    )

    ip_adapter_num_tokens: int = 4
    ip_adapter_cross_attn_dim: int = 2048


@dataclass
class TrainConfig:
    output_dir: str = "./checkpoints"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 100
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 200
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 42
    log_every: int = 25
    save_every: int = 1000


@dataclass
class DataConfig:
    """Single-view data under ``data_root/<PART>_PPP/``."""

    data_root: str = "./data"
    part_ids_file: Optional[str] = None
    num_workers: int = 4
    drop_last: bool = True
