from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetPaths:
    raw_root: Path
    out_root: Path
    image_size: int = 512


@dataclass
class TrainConfig:
    processed_root: Path
    pretrained_model: str
    controlnet_model: str
    output_dir: Path
    image_size: int = 512
    batch_size: int = 2
    epochs: int = 20
    lr: float = 1e-4
    num_workers: int = 4
    mixed_precision: str = "fp16"
    grad_accum_steps: int = 1
    seed: int = 42
    max_grad_norm: float = 1.0


@dataclass
class InferConfig:
    input_image: Path
    processed_root: Path
    pretrained_model: str
    controlnet_model: str
    lora_dir: Path
    output_dir: Path
    image_size: int = 512
    num_steps: int = 0
    inference_steps_per_frame: int = 30
    guidance_scale: float = 5.5
    seed: int = 123

