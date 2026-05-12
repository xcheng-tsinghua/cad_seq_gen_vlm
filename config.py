from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetPaths:
    raw_root: Path
    out_root: Path
    val_ratio: float = 0.1


@dataclass
class TrainConfig:
    raw_root: Path
    processed_root: Path | None
    output_dir: Path
    image_size: int = 384
    base_channels: int = 32
    batch_size: int = 8
    epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
    sd_model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    w_sd_latent: float = 0.2


@dataclass
class InferConfig:
    input_image: Path
    raw_root: Path
    processed_root: Path | None
    checkpoint: Path | None
    output_dir: Path
    num_steps: int = 0
    threshold: float = 0.5
    seed: int = 123
    device: str = "cuda"

