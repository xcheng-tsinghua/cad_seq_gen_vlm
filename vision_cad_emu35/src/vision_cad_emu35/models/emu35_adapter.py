from __future__ import annotations

import ast
import re
import sys
import warnings
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from PIL import Image

from vision_cad_emu35.config import ALLOWED_ATTN_IMPLEMENTATIONS, GenerationConfig, ModelConfig, resolve_project_path
from vision_cad_emu35.model_paths import DOWNLOAD_COMMAND, ensure_default_local_model_paths, validate_local_model_paths
from vision_cad_emu35.models.emu35_compat import apply_emu3_tokenizer_compat, is_special_tokens_set_error
from vision_cad_emu35.utils.gpu import get_gpu_info
from vision_cad_emu35.utils.image_io import resize_pad_image
from vision_cad_emu35.utils.runtime_env import normalize_thread_env


SPECIAL_TOKENS = {
    "BOS": "<|extra_203|>",
    "EOS": "<|extra_204|>",
    "PAD": "<|endoftext|>",
    "BSS": "<|extra_100|>",
    "ESS": "<|extra_101|>",
    "BOC": "<|extra_50|>",
    "EOC": "<|extra_51|>",
}

EMU35_SAMPLING_DEFAULTS = {
    "use_cache": True,
    "text_top_k": 0,
    "text_top_p": 0.9,
    "text_temperature": 0.2,
    "image_top_k": 2048,
    "image_top_p": 0.9,
    "image_temperature": 1.0,
    "top_k": 0,
    "top_p": 0.9,
    "temperature": 0.2,
    "max_new_tokens": 1024,
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "guidance_scale": 1.0,
    "use_differential_sampling": True,
}

EMU35_GENERATION_DEFAULTS = {
    "max_position_embeddings": 32768,
    "unconditional_type": "no_text",
    "classifier_free_guidance": 1.0,
    "guidance_scale": 1.0,
    "negative_prompt": "",
    "cfg_scale": 1.0,
    "image_cfg_scale": 1.0,
    "max_img_token": 4096,
    "stream": False,
    "streaming": False,
}

EMU35_REQUIRED_CFG_FIELDS = (
    "image_area",
    "sampling_params",
    "special_token_ids",
    "classifier_free_guidance",
    "unconditional_type",
    "target_height",
    "target_width",
    "image_cfg_scale",
)


class Emu35Adapter:
    """Frozen Emu3.5 inference adapter.

    This adapter is intentionally inference-only. It does not fine-tune, compute
    losses, call backward, or download model files. Any Emu3.5 API differences
    should be isolated in this module.
    """

    def __init__(self, config: ModelConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            known = set(ModelConfig.__dataclass_fields__)
            self.config = ModelConfig(**{k: v for k, v in config.items() if k in known})
            self.extra_config = dict(config)
        else:
            self.config = config
            self.extra_config = asdict(config)
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.vq_model: Any | None = None
        self.official: dict[str, Any] = {}
        self.device = None

    def load_model(self) -> None:
        normalize_thread_env()
        ensure_default_local_model_paths(self.config)
        validate_local_model_paths(self.config)
        self.extra_config.update(asdict(self.config))
        compat_report = apply_emu3_tokenizer_compat(self.config)
        self.extra_config["tokenizer_compat"] = compat_report.to_dict()
        self._prepare_import_path()
        official = self._import_official_utilities()
        if official is None:
            raise NotImplementedError(
                "Emu3.5 official utilities were not found. Install the BAAI Emu3.5 repo "
                "or set model.emu_repo_path to its checkout so imports like "
                "`from src.utils.model_utils import build_emu3p5` work."
            )
        self.official = official

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load frozen Emu3.5 for inference.") from exc

        model_path = self.config.model_id_or_path
        tokenizer_path = self.config.tokenizer_path or self.config.tokenizer_id_or_path or model_path
        vq_path = self.config.vision_tokenizer_path or self.config.vq_path
        if not vq_path:
            raise ValueError("Set model.vision_tokenizer_path to the local downloaded vision tokenizer directory.")

        device = self.config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model_device = self.config.device_map or self.extra_config.get("hf_device") or device
        vq_device = self.config.vq_device or device
        base_build_kwargs = dict(self.extra_config.get("diffusion_decoder_kwargs") or {})
        attn_impl = self._effective_attn_implementation()

        try:
            self.model, self.tokenizer, self.vq_model = self._build_emu3p5_with_attention(
                attn_impl,
                base_build_kwargs,
                model_path,
                tokenizer_path,
                vq_path,
                model_device,
                vq_device,
            )
        except Exception as exc:
            if is_special_tokens_set_error(exc):
                raise RuntimeError(
                    "Emu3Tokenizer failed during initialization because special_tokens_set is missing. "
                    "This is a custom tokenizer / transformers compatibility issue. "
                    "Run scripts/check_emu35_tokenizer.py and ensure the tokenizer compatibility patch is applied. "
                    f"Original {type(exc).__name__}: {exc}"
                ) from exc
            if self._is_flash_attention_2_error(exc):
                message = "Flash Attention 2 is not supported by Emu3ForCausalLM. Falling back to eager attention."
                print(f"WARNING: {message}", file=sys.stderr)
                warnings.warn(message, RuntimeWarning)
                try:
                    self.model, self.tokenizer, self.vq_model = self._build_emu3p5_with_attention(
                        "eager",
                        base_build_kwargs,
                        model_path,
                        tokenizer_path,
                        vq_path,
                        model_device,
                        vq_device,
                    )
                    attn_impl = "eager"
                except Exception as retry_exc:
                    if is_special_tokens_set_error(retry_exc):
                        raise RuntimeError(
                            "Emu3Tokenizer failed during initialization because special_tokens_set is missing. "
                            "This is a custom tokenizer / transformers compatibility issue. "
                            "Run scripts/check_emu35_tokenizer.py and ensure the tokenizer compatibility patch is applied. "
                            f"Original {type(retry_exc).__name__}: {retry_exc}"
                        ) from retry_exc
                    raise RuntimeError(
                        "Failed to load Emu3.5 after retrying with attn_implementation='eager'. "
                        f"Original {type(exc).__name__}: {exc}. "
                        f"Retry {type(retry_exc).__name__}: {retry_exc}. "
                        f"If local weights are missing, run: {DOWNLOAD_COMMAND}."
                    ) from retry_exc
            else:
                raise RuntimeError(
                    "Failed to load Emu3.5. This may be an attention implementation issue, not a model-path issue. "
                    f"Configured attn_implementation={attn_impl!r}. "
                    f"Original {type(exc).__name__}: {exc}. "
                    f"If local weights are missing, run: {DOWNLOAD_COMMAND}."
                ) from exc
        self.extra_config["effective_attn_implementation"] = attn_impl
        self.device = getattr(self.model, "device", torch.device(device))
        if hasattr(self.model, "eval"):
            self.model.eval()

    def generate_multimodal(
        self,
        prompt_text: str,
        images: list[Image.Image],
        generation_config: GenerationConfig | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_loaded()
        runtime_cfg = self._runtime_cfg(generation_config)
        image_str = "".join(
            self.official["build_image"](
                resize_pad_image(image, self.config.image_size),
                runtime_cfg,
                self.tokenizer,
                self.vq_model,
            )
            for image in images
        )
        full_prompt = self._format_multimodal_prompt(prompt_text, image_str)
        unconditional_prompt = self._format_multimodal_prompt("You are a helpful CAD modeling assistant.", image_str)

        import torch

        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        unconditional_ids = self.tokenizer.encode(unconditional_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        text_parts: list[str] = []
        image: Image.Image | None = None
        try:
            with torch.no_grad():
                iterator = self.official["generate"](
                    runtime_cfg,
                    self.model,
                    self.tokenizer,
                    input_ids,
                    unconditional_ids,
                    None,
                    False,
                )
                for event in iterator:
                    decoded_text, decoded_image = self._decode_generation_event(event)
                    if decoded_text:
                        text_parts.append(decoded_text)
                    if decoded_image is not None:
                        image = decoded_image
        except AttributeError as exc:
            missing_field = _missing_attribute_name(exc)
            if missing_field and "SimpleNamespace" in str(exc):
                raise RuntimeError(
                    "Emu3.5 generation config is missing required field: "
                    f"{missing_field}. Please update configs/rag.yaml or build_emu35_generation_cfg()."
                ) from exc
            raise RuntimeError(f"Emu3.5 generation failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Emu3.5 generation failed: {exc}") from exc

        raw_text = "".join(text_parts).strip()
        return {
            "operation_type": self.parse_operation_type(raw_text),
            "image": resize_pad_image(image, self.config.image_size) if image is not None else None,
            "raw_text": raw_text,
            "metadata": {"gpu": get_gpu_info(), "num_prompt_images": len(images)},
        }

    def parse_operation_type(self, raw_text: str) -> str:
        return parse_operation_type(raw_text)

    def _format_multimodal_prompt(self, prompt_text: str, image_str: str) -> str:
        return (
            f"{SPECIAL_TOKENS['BOS']}{image_str}\n"
            f"USER:\n{prompt_text}\n"
            f"ASSISTANT: {SPECIAL_TOKENS['BSS']}"
        )

    def _runtime_cfg(self, generation_config: GenerationConfig | dict[str, Any] | None = None) -> Any:
        cfg = build_emu35_generation_cfg(generation_config)
        cfg.model_path = self.config.model_id_or_path
        cfg.tokenizer_path = self.config.tokenizer_path or self.config.model_id_or_path
        cfg.vq_path = self.config.vision_tokenizer_path or self.config.vq_path
        cfg.local_files_only = self.config.local_files_only
        cfg.vq_type = self.config.vq_type
        cfg.task_type = self.config.task_type
        cfg.use_image = True
        cfg.image_area = self.config.image_area or self.config.image_size * self.config.image_size
        cfg.target_height = self.config.image_size
        cfg.target_width = self.config.image_size
        cfg.vision_tokenizer = self.vq_model
        cfg.special_token_ids = self._special_token_ids()
        return cfg

    def _special_token_ids(self) -> dict[str, int]:
        ids: dict[str, int] = {}
        for name, token in SPECIAL_TOKENS.items():
            try:
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                if encoded:
                    ids[name] = int(encoded[0])
                    continue
            except Exception:
                pass
            value = getattr(self.tokenizer, f"{name.lower()}_token_id", None)
            if value is not None:
                ids[name] = int(value)
        if "PAD" not in ids:
            ids["PAD"] = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        if "EOS" not in ids:
            ids["EOS"] = int(getattr(self.tokenizer, "eos_token_id", ids["PAD"]))
        return ids

    def _decode_generation_event(self, event: Any) -> tuple[str, Image.Image | None]:
        if isinstance(event, dict):
            if event.get("type") == "text":
                return str(event.get("text", "")), None
            if event.get("type") == "image" and event.get("image") is not None:
                mm_out = self.official["multimodal_decode"](event["image"], self.tokenizer, self.vq_model)
                return self._extract_text_image_from_mm(mm_out)
            return "", None

        result = self.tokenizer.decode(event, skip_special_tokens=False)
        mm_out = self.official["multimodal_decode"](result, self.tokenizer, self.vq_model)
        return self._extract_text_image_from_mm(mm_out)

    def _extract_text_image_from_mm(self, mm_out: Any) -> tuple[str, Image.Image | None]:
        text_parts: list[str] = []
        image: Image.Image | None = None
        if isinstance(mm_out, list):
            for item in mm_out:
                if isinstance(item, Image.Image):
                    image = item
                elif isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, (tuple, list)) and item:
                    value = item[-1]
                    if isinstance(value, Image.Image):
                        image = value
                    elif isinstance(value, str):
                        text_parts.append(value)
        elif isinstance(mm_out, Image.Image):
            image = mm_out
        elif isinstance(mm_out, str):
            text_parts.append(mm_out)
        return "".join(text_parts), image

    def _prepare_import_path(self) -> None:
        repo_path = resolve_project_path(self.config.emu_repo_path)
        if repo_path:
            repo = str(repo_path)
            self.extra_config["resolved_emu_repo_path"] = repo
            if repo not in sys.path:
                sys.path.insert(0, repo)

    def _build_emu3p5_with_attention(
        self,
        attn_implementation: str,
        base_build_kwargs: dict[str, Any],
        model_path: str,
        tokenizer_path: str,
        vq_path: str,
        model_device: str,
        vq_device: str,
    ) -> tuple[Any, Any, Any]:
        build_fn = self.official["build_emu3p5"]
        build_kwargs = dict(base_build_kwargs)
        build_kwargs["attn_implementation"] = attn_implementation
        build_kwargs = self._sanitize_optional_acceleration_kwargs(build_kwargs)
        if self.config.local_files_only:
            build_kwargs["local_files_only"] = True
        build_kwargs = self._filter_supported_kwargs(build_fn, build_kwargs)

        with self._force_from_pretrained_attention(attn_implementation):
            return build_fn(
                model_path,
                tokenizer_path,
                vq_path,
                vq_type=self.config.vq_type,
                model_device=model_device,
                vq_device=vq_device,
                **build_kwargs,
            )

    def _effective_attn_implementation(self) -> str:
        value = str(getattr(self.config, "attn_implementation", "eager") or "eager").strip()
        if value not in ALLOWED_ATTN_IMPLEMENTATIONS:
            allowed = ", ".join(ALLOWED_ATTN_IMPLEMENTATIONS)
            raise ValueError(f"Unsupported attn_implementation={value!r}. Allowed values: {allowed}.")
        if value == "auto":
            return "eager"
        return value

    def _is_flash_attention_2_error(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return "flash attention 2" in text or "flash_attention_2" in text

    @contextmanager
    def _force_from_pretrained_attention(self, attn_implementation: str) -> Iterator[None]:
        patches: list[tuple[Any, str, Any]] = []

        def patch_class(target: Any) -> None:
            if not isinstance(target, type):
                return
            original_attr = target.__dict__.get("from_pretrained")
            if original_attr is None:
                return
            if any(existing[0] is target for existing in patches):
                return
            if isinstance(original_attr, classmethod):
                original_func = original_attr.__func__
            else:
                bound = getattr(target, "from_pretrained")
                original_func = getattr(bound, "__func__", bound)

            def patched(cls: Any, *args: Any, **kwargs: Any) -> Any:
                kwargs["attn_implementation"] = attn_implementation
                return original_func(cls, *args, **kwargs)

            setattr(target, "from_pretrained", classmethod(patched))
            patches.append((target, "from_pretrained", original_attr))

        try:
            from transformers import AutoModelForCausalLM
            from transformers.modeling_utils import PreTrainedModel

            patch_class(PreTrainedModel)
            patch_class(AutoModelForCausalLM)
        except Exception:
            pass

        build_fn = self.official.get("build_emu3p5")
        for value in getattr(build_fn, "__globals__", {}).values():
            name = getattr(value, "__name__", "")
            if isinstance(value, type) and ("Emu3" in name or name.endswith("ForCausalLM")):
                patch_class(value)

        try:
            yield
        finally:
            for target, name, original_attr in reversed(patches):
                setattr(target, name, original_attr)

    def _import_official_utilities(self) -> dict[str, Any] | None:
        try:
            from src.utils.generation_utils import generate, multimodal_decode
            from src.utils.input_utils import build_image
            from src.utils.model_utils import build_emu3p5
        except Exception:
            return None
        return {
            "build_emu3p5": build_emu3p5,
            "build_image": build_image,
            "generate": generate,
            "multimodal_decode": multimodal_decode,
        }

    def _maybe_add_supported_kwarg(self, fn: Any, kwargs: dict[str, Any], name: str, value: Any) -> dict[str, Any]:
        try:
            sig = signature(fn)
        except (TypeError, ValueError):
            return kwargs
        supports_var_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values())
        if supports_var_kwargs or name in sig.parameters:
            updated = dict(kwargs)
            updated.setdefault(name, value)
            return updated
        return kwargs

    def _filter_supported_kwargs(self, fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = signature(fn)
        except (TypeError, ValueError):
            return kwargs
        supports_var_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values())
        if supports_var_kwargs:
            return kwargs
        return {key: value for key, value in kwargs.items() if key in sig.parameters}

    def _sanitize_optional_acceleration_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        updated = dict(kwargs)
        optional_modules = {
            "flash_attn": ("flash_attn", "use_flash_attn", "flash_attention", "attn_implementation"),
            "xformers": ("xformers", "use_xformers"),
            "bitsandbytes": ("bitsandbytes", "load_in_4bit", "load_in_8bit", "quantization_config"),
            "vllm": ("vllm", "use_vllm"),
        }
        for module_name, keys in optional_modules.items():
            if not any(key in updated for key in keys):
                continue
            if module_name == "flash_attn":
                flash_requested = any(key in updated for key in ("use_flash_attn", "flash_attention"))
                flash_requested = flash_requested or updated.get("attn_implementation") == "flash_attention_2"
                if not flash_requested:
                    continue
            if self._module_available(module_name):
                continue
            for key in keys:
                if key in updated:
                    if key == "attn_implementation":
                        updated[key] = "eager"
                    elif key == "quantization_config":
                        updated.pop(key, None)
                    else:
                        updated[key] = False
            warnings.warn(
                f"Optional acceleration library {module_name!r} is not installed; falling back to standard PyTorch inference.",
                RuntimeWarning,
            )
        return updated

    def _module_available(self, module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except Exception:
            return False

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self.vq_model is None:
            raise RuntimeError("Emu35Adapter.load_model() must be called before generation.")


def parse_operation_type(raw_text: str) -> str:
    match = re.search(r"Operation_Type\s*:\s*([A-Za-z0-9_<>\-_]+)", raw_text)
    if match:
        return match.group(1).strip()
    stripped = raw_text.strip()
    if stripped.startswith("<STOP>"):
        return "<STOP>"
    first_line = stripped.splitlines()[0] if stripped else ""
    return first_line.strip() or "other"


def build_emu35_generation_cfg(project_generation_config: GenerationConfig | dict[str, Any] | None) -> SimpleNamespace:
    """Build the config object expected by official Emu3.5 generation utilities."""

    raw = _generation_config_to_dict(project_generation_config)
    sampling_params = dict(EMU35_SAMPLING_DEFAULTS)
    for key in tuple(sampling_params):
        if key in raw and raw[key] is not None:
            sampling_params[key] = raw[key]
    sampling_params["max_new_tokens"] = raw.get("max_new_tokens", sampling_params["max_new_tokens"])
    sampling_params["top_p"] = raw.get("top_p", sampling_params["top_p"])
    sampling_params["temperature"] = raw.get("temperature", sampling_params["temperature"])
    sampling_params["top_k"] = raw.get("top_k", sampling_params["top_k"])
    sampling_params["do_sample"] = raw.get("do_sample", sampling_params["do_sample"])
    sampling_params["num_beams"] = raw.get("num_beams", sampling_params["num_beams"])
    sampling_params["repetition_penalty"] = raw.get("repetition_penalty", sampling_params["repetition_penalty"])
    sampling_params["length_penalty"] = raw.get("length_penalty", sampling_params["length_penalty"])
    sampling_params["use_cache"] = raw.get("use_cache", sampling_params["use_cache"])
    sampling_params["guidance_scale"] = raw.get("guidance_scale", sampling_params["guidance_scale"])
    sampling_params["use_differential_sampling"] = raw.get(
        "use_differential_sampling",
        sampling_params["use_differential_sampling"],
    )

    cfg = SimpleNamespace()
    for key, default in EMU35_GENERATION_DEFAULTS.items():
        setattr(cfg, key, raw.get(key, default))
    cfg.sampling_params = sampling_params
    cfg.streaming = bool(raw.get("streaming", raw.get("stream", cfg.streaming)))
    cfg.stream = cfg.streaming
    cfg.max_new_tokens = sampling_params["max_new_tokens"]
    cfg.max_image_tokens = raw.get("max_image_tokens")
    cfg.max_position_embeddings = raw.get("max_position_embeddings", cfg.max_position_embeddings)
    cfg.image_area = raw.get("image_area", 512 * 512)
    cfg.target_height = raw.get("target_height", 512)
    cfg.target_width = raw.get("target_width", 512)
    cfg.special_token_ids = raw.get("special_token_ids", {})
    cfg.vision_tokenizer = raw.get("vision_tokenizer")

    for key, value in sampling_params.items():
        setattr(cfg, key, value)
    return cfg


def inspect_generation_utils_cfg_fields(path: str | Path) -> dict[str, list[str]]:
    """Statically inspect official generation_utils.py for cfg field reads."""

    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source_path))
    attrs: set[str] = set()
    getattr_attrs: set[str] = set()
    sampling_keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "cfg":
            attrs.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "cfg"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            getattr_attrs.add(node.args[1].value)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "cfg"
            and node.value.attr == "sampling_params"
        ):
            key = _literal_subscript_key(node)
            if key is not None:
                sampling_keys.add(key)
    return {
        "cfg_attributes": sorted(attrs | getattr_attrs),
        "cfg_getattr_attributes": sorted(getattr_attrs),
        "sampling_param_keys": sorted(sampling_keys),
    }


def _generation_config_to_dict(config: GenerationConfig | dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return asdict(GenerationConfig())
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _literal_subscript_key(node: ast.Subscript) -> str | None:
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def _missing_attribute_name(exc: AttributeError) -> str | None:
    match = re.search(r"has no attribute '([^']+)'", str(exc))
    return match.group(1) if match else None
