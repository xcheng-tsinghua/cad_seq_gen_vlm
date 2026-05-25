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
    tokenizer_path: str | None = _DEFAULT_MODEL_PATHS["tokenizer_path"]
    vision_tokenizer_path: str | None = _DEFAULT_MODEL_PATHS["vision_tokenizer_path"]
    emu_repo_path: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = True
    image_size: int = 512
    precision: str = "bf16"
    device_map: str | None = "auto"
    tokenizer_id_or_path: str | None = None
    vq_path: str | None = None
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
    optional_adapter_path: str | None = None


@dataclass
class DataConfig:
    dataset_root: str = "/path/to/your/dataset"
    manifest_dir: str = "data/manifests"
    output_dir: str = "outputs"
    add_stop_samples: bool = True
    stop_image_policy: str = "copy_last_depth"
    image_size: int = 512


@dataclass
class RagConfig:
    kb_dir: str = "/root/autodl-tmp/data/outputs/rag_kb"
    embedding_backend: str = "simple"
    vector_backend: str = "numpy"
    top_k: int = 3
    max_reference_images: int = 6
    include_example_outputs: bool = True
    include_example_final_snapshot: bool = False
    include_example_prev_depth_map: bool = True
    include_example_overlayed_all: bool = True
    allow_empty_kb: bool = True
    operation_type_hint: str | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    max_image_tokens: int | None = None
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    artifacts_dir: str = "outputs/api_artifacts"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @property
    def artifact_dir(self) -> str:
        return self.artifacts_dir

    @property
    def allow_origins(self) -> list[str]:
        return self.cors_origins


@dataclass
class WebConfig:
    title: str = "Vision CAD Emu3.5 RAG Demo"
    allow_upload_kb: bool = False
    show_retrieved_examples: bool = True


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        return cls(
            model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
            data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
            rag=_dataclass_from_dict(RagConfig, raw.get("rag", {})),
            generation=_dataclass_from_dict(GenerationConfig, raw.get("generation", {})),
            api=_dataclass_from_dict(ApiConfig, raw.get("api", {})),
            web=_dataclass_from_dict(WebConfig, raw.get("web", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataclass_from_dict(cls: type[Any], raw: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    normalized = dict(raw or {})
    if cls is ApiConfig:
        if "artifact_dir" in normalized and "artifacts_dir" not in normalized:
            normalized["artifacts_dir"] = normalized.pop("artifact_dir")
        if "allow_origins" in normalized and "cors_origins" not in normalized:
            normalized["cors_origins"] = normalized.pop("allow_origins")
    return cls(**{k: v for k, v in normalized.items() if k in names})


def load_config(path: str | Path | None) -> AppConfig:
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = config.to_dict() if is_dataclass(config) else config
    try:
        import yaml

        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    except ImportError:
        import json

        target.write_text(json.dumps(raw, indent=2), encoding="utf-8")
