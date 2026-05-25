from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from vision_cad_emu35.config import GenerationConfig, ModelConfig
from vision_cad_emu35.data.dataset import SYSTEM_PROMPT, USER_PROMPT
from vision_cad_emu35.model_paths import DOWNLOAD_COMMAND, ensure_default_local_model_paths, validate_local_model_paths
from vision_cad_emu35.utils.gpu import get_gpu_info
from vision_cad_emu35.utils.image_io import resize_pad_image


SPECIAL_TOKENS = {
    "BOS": "<|extra_203|>",
    "EOS": "<|extra_204|>",
    "PAD": "<|endoftext|>",
    "EOL": "<|extra_200|>",
    "EOF": "<|extra_201|>",
    "TMS": "<|extra_202|>",
    "IMG": "<|image token|>",
    "BOI": "<|image start|>",
    "EOI": "<|image end|>",
    "BSS": "<|extra_100|>",
    "ESS": "<|extra_101|>",
    "BOG": "<|extra_60|>",
    "EOG": "<|extra_61|>",
    "BOC": "<|extra_50|>",
    "EOC": "<|extra_51|>",
}


class Emu35Adapter:
    """Single boundary for all Emu3.5-specific model, tokenizer, and image-token logic.

    The public Emu3.5 repository exposes inference through utilities such as
    ``build_emu3p5``, ``build_image``, ``generate``, and ``multimodal_decode``.
    This adapter uses those utilities when available. If your installed Emu3.5
    package exposes a different API, update this file only.
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
        self.pad_token_id: int = 0

    def load_model(self) -> None:
        """Load Emu3.5 through the official repo utilities if available."""
        ensure_default_local_model_paths(self.config)
        validate_local_model_paths(self.config)
        self.extra_config.update(asdict(self.config))
        self._prepare_import_path()
        official = self._import_official_utilities()
        if official is None:
            raise NotImplementedError(
                "Emu3.5 official utilities were not found. Install the BAAI Emu3.5 repo "
                "or set model.emu_repo_path to its checkout so imports like "
                "`from src.utils.model_utils import build_emu3p5` work. The rest of the "
                "codebase is ready; only this adapter needs to match your Emu3.5 install."
            )
        self.official = official
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load Emu3.5.") from exc

        model_path = self.config.model_id_or_path
        tokenizer_path = self.extra_config.get("tokenizer_path") or self.extra_config.get("tokenizer_id_or_path") or model_path
        vq_path = self.extra_config.get("vision_tokenizer_path") or self.extra_config.get("vq_path")
        if not vq_path:
            raise ValueError(
                "Emu3.5 requires a vision tokenizer. Set model.vision_tokenizer_path "
                "or model.vq_path to the local downloaded vision tokenizer directory."
            )

        device = self.extra_config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model_device = self.config.device_map or self.extra_config.get("hf_device") or device
        vq_device = self.extra_config.get("vq_device") or device
        build_kwargs = dict(self.extra_config.get("diffusion_decoder_kwargs") or {})
        if self.config.local_files_only:
            build_kwargs = self._maybe_add_supported_kwarg(
                official["build_emu3p5"],
                build_kwargs,
                "local_files_only",
                True,
            )

        try:
            self.model, self.tokenizer, self.vq_model = official["build_emu3p5"](
                model_path,
                tokenizer_path,
                vq_path,
                vq_type=self.extra_config.get("vq_type", "ibq"),
                model_device=model_device,
                vq_device=vq_device,
                **build_kwargs,
            )
        except Exception as exc:
            if self.config.local_files_only:
                raise RuntimeError(
                    "Failed to load Emu3.5 from local files. Confirm the local paths in the config "
                    f"or run: {DOWNLOAD_COMMAND}. Original error: {exc}"
                ) from exc
            raise
        self.device = getattr(self.model, "device", torch.device(device))
        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if self.config.use_lora or self.config.use_qlora:
            self._apply_lora()
        self.pad_token_id = self._token_id(SPECIAL_TOKENS["PAD"]) or getattr(self.tokenizer, "pad_token_id", 0) or 0

    def preprocess_inputs(
        self,
        final_snapshot: Image.Image,
        prev_depth_map: Image.Image,
        prompt: str | None,
    ) -> dict[str, Any]:
        return {
            "final_snapshot": resize_pad_image(final_snapshot, self.config.image_size),
            "prev_depth_map": resize_pad_image(prev_depth_map, self.config.image_size),
            "prompt": prompt or USER_PROMPT,
            "system_prompt": SYSTEM_PROMPT,
        }

    def build_training_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Build a supervised next-token sample for native multimodal autoregression.

        This uses the official vision tokenizer to encode both input images and the
        target preview image into Emu3.5 image-token text. Labels are masked over
        the prompt and active over assistant text plus target image tokens.
        """
        self._require_loaded()
        if not self.official.get("build_image"):
            raise NotImplementedError("Official build_image utility is required for multimodal target construction.")

        cfg = self._runtime_cfg()
        final_image = sample["final_snapshot"]
        prev_image = sample["prev_depth_map"]
        target_image = sample["target_image"]
        prompt = sample.get("prompt") or USER_PROMPT
        operation_type = sample["operation_type"]

        image_str = (
            self.official["build_image"](final_image, cfg, self.tokenizer, self.vq_model)
            + self.official["build_image"](prev_image, cfg, self.tokenizer, self.vq_model)
        )
        target_image_str = self.official["build_image"](target_image, cfg, self.tokenizer, self.vq_model)
        prompt_text = self._format_prompt(prompt).replace("<|IMAGE|>", image_str)

        # TODO(Emu3.5 adapter): verify whether the installed release expects EOC,
        # ESS, or another delimiter between generated text and generated image.
        target_text = f"Operation_Type: {operation_type}{SPECIAL_TOKENS['EOC']}{target_image_str}"

        import torch

        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)[0]
        target_ids = self.tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False)[0]
        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "metadata": {k: v for k, v in sample.items() if isinstance(v, (str, int, float, bool, type(None)))},
        }

    def build_training_batch(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        self._require_loaded()
        import torch
        from torch.nn.utils.rnn import pad_sequence

        encoded = [self.build_training_sample(sample) for sample in samples]
        input_ids = pad_sequence([item["input_ids"] for item in encoded], batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence([item["labels"] for item in encoded], batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            "input_ids": input_ids.to(self.device),
            "labels": labels.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "metadata": [item["metadata"] for item in encoded],
        }

    def forward_loss(self, batch: dict[str, Any]) -> Any:
        self._require_loaded()
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        if not hasattr(outputs, "loss"):
            raise RuntimeError("Emu3.5 model forward did not return a .loss attribute.")
        return outputs.loss

    def generate(
        self,
        final_snapshot: Image.Image,
        prev_depth_map: Image.Image,
        prompt: str | None,
        generation_config: GenerationConfig | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate operation type and preview image."""
        self._require_loaded()
        gen_cfg = generation_config if isinstance(generation_config, GenerationConfig) else GenerationConfig(**(generation_config or {}))
        runtime_cfg = self._runtime_cfg(gen_cfg)
        prepared = self.preprocess_inputs(final_snapshot, prev_depth_map, prompt)
        image_str = (
            self.official["build_image"](prepared["final_snapshot"], runtime_cfg, self.tokenizer, self.vq_model)
            + self.official["build_image"](prepared["prev_depth_map"], runtime_cfg, self.tokenizer, self.vq_model)
        )
        prompt_text = self._format_prompt(prepared["prompt"]).replace("<|IMAGE|>", image_str)
        unc_prompt = self._unconditional_prompt().replace("<|IMAGE|>", image_str)

        import torch

        input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        bos_id = self._token_id(SPECIAL_TOKENS["BOS"])
        if bos_id is not None and input_ids.numel() and int(input_ids[0, 0]) != bos_id:
            input_ids = torch.cat([torch.tensor([[bos_id]], device=input_ids.device, dtype=input_ids.dtype), input_ids], dim=1)
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

        text_parts: list[str] = []
        image: Image.Image | None = None
        raw_events: list[str] = []
        try:
            iterator = self.official["generate"](runtime_cfg, self.model, self.tokenizer, input_ids, unconditional_ids, None, False)
            for event in iterator:
                decoded_text, decoded_image = self._decode_generation_event(event)
                if decoded_text:
                    text_parts.append(decoded_text)
                    raw_events.append(decoded_text)
                if decoded_image is not None:
                    image = decoded_image
        except Exception as exc:
            raise RuntimeError(f"Emu3.5 generation failed: {exc}") from exc

        raw_text = "".join(text_parts).strip()
        if image is None:
            raise NotImplementedError(
                "Generation completed but no decoded image was returned. Update "
                "Emu35Adapter._decode_generation_event to match your installed "
                "Emu3.5 multimodal_decode output format."
            )
        return {
            "operation_type": parse_operation_type(raw_text),
            "image": resize_pad_image(image, self.config.image_size),
            "raw_text": raw_text,
            "metadata": {
                "gpu": get_gpu_info(),
                "raw_event_count": len(raw_events),
            },
        }

    def save_checkpoint(self, output_dir: str | Path) -> None:
        self._require_loaded()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out)
        else:
            import torch

            torch.save(self.model.state_dict(), out / "pytorch_model.bin")
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(out)
        (out / "adapter_config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        self._require_loaded()
        ckpt = Path(checkpoint_dir)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt}")
        try:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, ckpt)
            return
        except Exception:
            pass
        model_bin = ckpt / "pytorch_model.bin"
        if model_bin.exists():
            import torch

            state = torch.load(model_bin, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            return
        raise FileNotFoundError(f"No supported checkpoint payload found in {ckpt}")

    def _prepare_import_path(self) -> None:
        repo_path = self.extra_config.get("emu_repo_path")
        if repo_path:
            repo = str(Path(repo_path).resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

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

    def _runtime_cfg(self, generation_config: GenerationConfig | None = None) -> Any:
        gen = generation_config or GenerationConfig()
        cfg = SimpleNamespace()
        cfg.model_path = self.config.model_id_or_path
        cfg.tokenizer_path = self.extra_config.get("tokenizer_path") or self.config.model_id_or_path
        cfg.vq_path = self.extra_config.get("vision_tokenizer_path") or self.extra_config.get("vq_path")
        cfg.local_files_only = self.config.local_files_only
        cfg.vq_type = self.extra_config.get("vq_type", "ibq")
        cfg.task_type = self.extra_config.get("task_type", "x2i")
        cfg.use_image = True
        cfg.image_area = self.extra_config.get("image_area", self.config.image_size * self.config.image_size)
        cfg.target_height = self.config.image_size
        cfg.target_width = self.config.image_size
        cfg.streaming = False
        cfg.classifier_free_guidance = self.extra_config.get("classifier_free_guidance", 1.0)
        cfg.special_tokens = dict(SPECIAL_TOKENS)
        cfg.special_token_ids = {k: self._token_id(v) for k, v in SPECIAL_TOKENS.items()}
        cfg.sampling_params = {
            "use_cache": True,
            "text_top_k": self.extra_config.get("text_top_k", 1024),
            "text_top_p": gen.top_p,
            "text_temperature": gen.temperature,
            "image_top_k": self.extra_config.get("image_top_k", 5120),
            "image_top_p": self.extra_config.get("image_top_p", 1.0),
            "image_temperature": self.extra_config.get("image_temperature", 1.0),
            "top_k": self.extra_config.get("top_k", 131072),
            "top_p": gen.top_p,
            "temperature": gen.temperature,
            "num_beams_per_group": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "max_new_tokens": gen.max_new_tokens,
            "guidance_scale": self.extra_config.get("guidance_scale", 1.0),
            "use_differential_sampling": self.extra_config.get("use_differential_sampling", True),
            "do_sample": gen.do_sample,
            "num_beams": 1,
        }
        for key, value in cfg.sampling_params.items():
            setattr(cfg, key, value)
        return cfg

    def _format_prompt(self, user_prompt: str) -> str:
        return (
            f"{SPECIAL_TOKENS['BOS']}{SYSTEM_PROMPT}\n"
            f"USER: <|IMAGE|>{user_prompt}\n"
            f"ASSISTANT: {SPECIAL_TOKENS['BSS']}"
        )

    def _unconditional_prompt(self) -> str:
        return f"{SPECIAL_TOKENS['BOS']}You are a helpful assistant.\nUSER: <|IMAGE|>\nASSISTANT: {SPECIAL_TOKENS['BSS']}"

    def _decode_generation_event(self, event: Any) -> tuple[str, Image.Image | None]:
        if isinstance(event, dict):
            if event.get("type") == "text":
                return str(event.get("text", "")), None
            if event.get("type") == "image":
                image_token_str = event.get("image")
                if image_token_str is None:
                    return "", None
                mm_out = self.official["multimodal_decode"](image_token_str, self.tokenizer, self.vq_model)
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
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    kind = str(item[0]).lower()
                    value = item[-1]
                    if kind in {"text", "answer", "gen_text"}:
                        text_parts.append(str(value))
                    elif kind in {"image", "gen_image"} and isinstance(value, Image.Image):
                        image = value
                    elif isinstance(value, Image.Image):
                        image = value
                    elif isinstance(value, str):
                        text_parts.append(value)
                elif isinstance(item, Image.Image):
                    image = item
                elif isinstance(item, str):
                    text_parts.append(item)
        elif isinstance(mm_out, Image.Image):
            image = mm_out
        elif isinstance(mm_out, str):
            text_parts.append(mm_out)
        return "".join(text_parts), image

    def _token_id(self, token: str) -> int | None:
        if self.tokenizer is None:
            return None
        try:
            return int(self.tokenizer.convert_tokens_to_ids(token))
        except Exception:
            try:
                return int(self.tokenizer.encode(token)[0])
            except Exception:
                return None

    def _apply_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError("peft is required for LoRA/QLoRA. Install peft or set model.use_lora=false.") from exc

        if self.config.use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
        target_modules = self.extra_config.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self.vq_model is None:
            raise RuntimeError("Emu35Adapter.load_model() must be called before using the adapter.")


def parse_operation_type(raw_text: str) -> str:
    match = re.search(r"Operation_Type\s*:\s*([A-Za-z0-9_<>\-_]+)", raw_text)
    if match:
        return match.group(1).strip()
    stripped = raw_text.strip()
    if stripped.startswith("<STOP>"):
        return "<STOP>"
    first_line = stripped.splitlines()[0] if stripped else ""
    return first_line.strip() or "other"
