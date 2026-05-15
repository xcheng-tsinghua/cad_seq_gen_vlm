"""Global configuration for the CAD multi-view sequence generator.

All shape-related constants live here so that `dataset.py`, the custom
ControlNet, the pipeline and the training / inference scripts stay in sync.

Layout of one modeling step (``G_k``):

    Rows (NUM_ROWS = 4)        -> components of the modeling step
        0: Prev Depth Map
        1: Sketch Plane Mask
        2: Reference Mask
        3: Result Wireframe

    Columns (NUM_VIEWS = 8)   -> camera view angles
        0: V3d_XposYposZpos
        1: V3d_XposYposZneg
        2: V3d_XposYnegZpos
        3: V3d_XposYnegZneg
        4: V3d_XnegYposZpos
        5: V3d_XnegYposZneg
        6: V3d_XnegYnegZpos
        7: V3d_XnegYnegZneg

Therefore each grid tensor has shape ``(3, NUM_ROWS * TILE_H, NUM_VIEWS * TILE_W)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Where every HuggingFace download (Qwen2.5-VL, SDXL, CLIP-Vision, etc.)
# should land. The folder lives next to this file so the project is fully
# self-contained on disk: no surprise downloads into ``~/.cache/huggingface``.
# ---------------------------------------------------------------------------
PRETRAINED_DIR: str = str((Path(__file__).resolve().parent / "pretrained_lm").resolve())


def set_hf_cache_env(force: bool = False) -> str:
    """Point HuggingFace's download caches at :data:`PRETRAINED_DIR`.

    Sets ``HF_HOME``, ``HF_HUB_CACHE``, ``HUGGINGFACE_HUB_CACHE`` and the
    legacy ``TRANSFORMERS_CACHE`` env vars. By default a variable is left
    untouched if the user has already exported it; pass ``force=True`` to
    override.

    Must be called BEFORE the first ``import transformers`` /
    ``import diffusers`` for the env vars to take effect. Because this
    function is invoked at the bottom of this module, importing ``config``
    anywhere in the project is sufficient.

    Returns
    -------
    str
        The resolved cache directory (== ``PRETRAINED_DIR``).
    """
    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    # Modern names: hub cache lives directly here (no ``hub/`` subfolder).
    cache_keys = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
    for k in cache_keys:
        if force or not os.environ.get(k):
            os.environ[k] = PRETRAINED_DIR

    # Parent dir used by some HF tools to derive other paths.
    if force or not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = PRETRAINED_DIR
    return PRETRAINED_DIR


# Auto-redirect on import so any downstream module that imports ``config``
# (which is every entry-point script in this project) gets the right cache
# location before it touches transformers / diffusers / huggingface_hub.
set_hf_cache_env()


# Row / view layout of the 4 x 8 grid =====================================
NUM_ROWS: int = 4
NUM_VIEWS: int = 8

ROW_NAMES: Tuple[str, ...] = (
    "prev_depth",
    "sketch_plane_mask",
    "ref_mask",
    "result_wireframe",
)

# Filenames on disk corresponding 1-to-1 with ``ROW_NAMES``. Decoupled from
# the semantic labels so that, e.g., switching dataset versions only requires
# touching this tuple.
ROW_FILENAMES: Tuple[str, ...] = (
    "prev_depth_map.png",
    "sketch_plane_mask.png",
    "reference_mask.png",
    "result_frame.png",
)

VIEW_NAMES: Tuple[str, ...] = (
    "V3d_XposYposZpos",
    "V3d_XposYposZneg",
    "V3d_XposYnegZpos",
    "V3d_XposYnegZneg",
    "V3d_XnegYposZpos",
    "V3d_XnegYposZneg",
    "V3d_XnegYnegZpos",
    "V3d_XnegYnegZneg",
)

# Short suffixes used in the on-disk folder names ``[CAD_PART_ID]_<SUFFIX>``.
# Maps to ``VIEW_NAMES`` element-wise. Encoding convention:
#     letter 1 -> X axis (P = positive, N = negative)
#     letter 2 -> Y axis
#     letter 3 -> Z axis
VIEW_SUFFIXES: Tuple[str, ...] = (
    "PPP",   # V3d_XposYposZpos
    "PPN",   # V3d_XposYposZneg
    "PNP",   # V3d_XposYnegZpos
    "PNN",   # V3d_XposYnegZneg
    "NPP",   # V3d_XnegYposZpos
    "NPN",   # V3d_XnegYposZneg
    "NNP",   # V3d_XnegYnegZpos
    "NNN",   # V3d_XnegYnegZneg
)

# Per-view canonical filenames inside each ``[CAD_PART_ID]_<SUFFIX>`` folder.
I_FINAL_FILENAME: str = "final_snapshot.png"
FINAL_SHAPE_FILENAME: str = "final_shape_frame.png"
STEP_DIR_PREFIX: str = "roll_back_index_"
PROMPT_FILENAME: str = "prompt.txt"

# Default "anchor" view used to look up step-level metadata (prompts) that is
# logically view-independent. Any of the 8 suffixes would work; we pick PPP.
ANCHOR_VIEW_SUFFIX: str = "PPP"

# Per-tile resolution. The full grid is NUM_ROWS*TILE_H by NUM_VIEWS*TILE_W.
TILE_H: int = 256
TILE_W: int = 256

# Single conditioning image (`I_final`) resolution. We keep it square to make
# the CLIP-Vision encoder happy without extra preprocessing.
IFINAL_H: int = 512
IFINAL_W: int = 512


@dataclass
class ModelConfig:
    """Hyperparameters describing the base diffusion model + adapters."""

    # Base text-to-image backbone. Swap to "black-forest-labs/FLUX.1-dev" for
    # the DiT variant; the rest of the framework is backbone-agnostic.
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # CLIP vision tower used by the IP-Adapter (global 3D reference branch).
    clip_image_encoder_name_or_path: str = "openai/clip-vit-large-patch14"

    # LoRA injected into UNet attention layers (queries + values + out + ff).
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    )

    # IP-Adapter image projector.
    ip_adapter_num_tokens: int = 4         # how many "image tokens" we mint
    ip_adapter_cross_attn_dim: int = 2048  # SDXL cross-attn dim; 768 for SD-1.5

    # Multi-view ControlNet.
    mv_cn_block_out_channels: Tuple[int, ...] = (16, 32, 96, 256)
    mv_cn_inner_channels: int = 320        # must match UNet `block_out_channels[0]`
    mv_cn_num_attn_heads: int = 8


@dataclass
class TrainConfig:
    """Optimization hyperparameters."""

    output_dir: str = "./checkpoints"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 100
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 200
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"          # "no" | "fp16" | "bf16"
    seed: int = 42

    # Snapshot / logging cadence.
    log_every: int = 25
    save_every: int = 1000


@dataclass
class DataConfig:
    """Where the on-disk dataset lives.

    The dataset is *self-describing*: ``CADMultiViewDataset`` auto-discovers
    ``[CAD_PART_ID]_<SUFFIX>`` folders under ``data_root``. The optional
    ``part_ids_file`` (a plain text file with one ``CAD_PART_ID`` per line)
    is only used to restrict the scan to a specific train/val split.
    """

    data_root: str = "./data"
    # Optional whitelist of CAD part IDs (one per line). ``None`` => use all.
    part_ids_file: Optional[str] = None
    # Sampling behaviour: pick a random view's `final_snapshot.png` as `I_final`
    # at each ``__getitem__`` call.
    random_i_final_view: bool = True
    # Tolerate parts that don't have all 8 view folders (skip them with a warning).
    require_all_views: bool = True
    num_workers: int = 4
    drop_last: bool = True
