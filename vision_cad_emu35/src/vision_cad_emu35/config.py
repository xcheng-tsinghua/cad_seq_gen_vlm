from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from vision_cad_emu35.model_paths import DEFAULT_MODEL_ROOT, default_local_model_paths


_DEFAULT_MODEL_PATHS = default_local_model_paths(DEFAULT_MODEL_ROOT)


@dataclass
class ModelConfig:
    model_root: str = DEFAULT_MODEL_ROOT
    model_id_or_path: str = _DEFAULT_MODEL_PATHS["model_id_or_path"]
    trust_remote_code: bool = True
    image_size: int = 512
    precision: str = "bf16"
    quantization: str | None = None
    device_map: str | None = "auto"
    tokenizer_path: str | None = _DEFAULT_MODEL_PATHS["tokenizer_path"]
    tokenizer_id_or_path: str | None = None
    vision_tokenizer_path: str | None = _DEFAULT_MODEL_PATHS["vision_tokenizer_path"]
    vq_path: str | None = None
    emu_repo_path: str | None = None
    local_files_only: bool = True
    vq_type: str = "ibq"
    device: str | None = None
    vq_device: str | None = None
    task_type: str = "x2i"
    image_area: int | None = None
    classifier_free_guidance: float = 1.0
    guidance_scale: float = 1.0
    text_top_k: int = 1024
    image_top_k: int = 5120
    image_top_p: float = 1.0
    image_temperature: float = 1.0
    top_k: int = 131072
    use_differential_sampling: bool = True
    use_lora: bool = True
    use_qlora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    dataset_root: str = "data/raw"
    output_dir: str = "outputs/emu35_finetune"
    manifest_dir: str = "data/manifests"
    add_stop_samples: bool = True
    stop_image_policy: str = "copy_last_depth"
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    split_by_part_id: bool = True
    image_size: int = 512
    num_workers: int = 8
    preprocessed_cache_dir: str | None = None


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 10
    batch_size_per_device: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    validation_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 5
    resume_from_checkpoint: str | None = None
    mixed_precision: str = "bf16"
    compile_model: bool = False
    max_text_length: int = 2048


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    max_image_tokens: int | None = None
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    wandb: bool = False
    project_name: str = "vision_cad_emu35"


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    artifact_dir: str = "outputs/api"
    allow_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        return cls(
            model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
            data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
            training=_dataclass_from_dict(TrainingConfig, raw.get("training", {})),
            generation=_dataclass_from_dict(GenerationConfig, raw.get("generation", {})),
            logging=_dataclass_from_dict(LoggingConfig, raw.get("logging", {})),
            api=_dataclass_from_dict(ApiConfig, raw.get("api", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataclass_from_dict(cls: type[Any], raw: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def load_config(path: str | Path | None) -> AppConfig:
    """Load YAML or JSON config into strongly typed dataclasses."""
    if path is None:
        return AppConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML config files.") from exc

        raw = yaml.safe_load(text) or {}
    else:
        import json

        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return AppConfig.from_dict(raw)


def save_config(config: AppConfig, path: str | Path) -> None:
    """Write config as YAML when PyYAML is available, otherwise JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = config.to_dict() if is_dataclass(config) else config
    try:
        import yaml

        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    except ImportError:
        import json

        target.write_text(json.dumps(raw, indent=2), encoding="utf-8")
