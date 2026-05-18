"""SDXL training/inference — single view, standard ControlNet, IP-Adapter.

// MVP Refactor: ``MultiViewControlNet`` / cross-view attention removed.
Uses ``diffusers.ControlNetModel`` + UNet (LoRA) + CLIP IP projection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from config import PRETRAINED_DIR, ModelConfig


class ImageProjModel(nn.Module):
    """IP-Adapter image token projector (SDXL cross-attn dim)."""

    def __init__(
        self,
        cross_attention_dim: int,
        clip_embeddings_dim: int = 1024,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Linear(clip_embeddings_dim, num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        b = image_embeds.shape[0]
        x = self.proj(image_embeds).reshape(b, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


class IPAttnProcessor(nn.Module):
    """Concat text + IP tokens in cross-attention (trainable K/V IP branch)."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 4,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        b, _, _ = hidden_states.shape
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            text_kv, ip_kv = encoder_hidden_states, None
        else:
            text_kv = encoder_hidden_states[:, : -self.num_tokens, :]
            ip_kv = encoder_hidden_states[:, -self.num_tokens :, :]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        key = attn.to_k(text_kv)
        value = attn.to_v(text_kv)

        head_dim = query.shape[-1] // attn.heads
        query = query.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(b, -1, attn.heads, head_dim).transpose(1, 2)

        text_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if ip_kv is not None:
            ip_key = self.to_k_ip(ip_kv)
            ip_value = self.to_v_ip(ip_kv)
            ip_key = ip_key.view(b, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(b, -1, attn.heads, head_dim).transpose(1, 2)
            ip_out = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            out = text_out + self.scale * ip_out
        else:
            out = text_out

        out = out.transpose(1, 2).reshape(b, -1, attn.heads * head_dim)
        out = out.to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(b, c, h, w)

        if attn.residual_connection:
            out = out + residual

        return out / attn.rescale_output_factor


@dataclass
class CADPipelineOutput:
    images: List[Image.Image]
    latents: torch.Tensor


class CADSingleViewPipeline(nn.Module):
    """Trainer/inference bundle aligned with ``StableDiffusionXLControlNetPipeline``.

    Loads ``ControlNetModel``, SDXL UNet/VAE/schedulers, dual text encoders, and a CLIP image
    encoder + small projection for IP-Adapter tokens. ``training_step_loss`` implements the same
    conditional denoising objective as the stock pipeline, without instantiating the
    ``DiffusionPipeline`` wrapper class.

    // MVP Refactor: replaces ``CADMultiViewPipeline``.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        device: Union[str, torch.device] = "cuda",
        weight_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype

        mid = model_cfg.pretrained_model_name_or_path
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            mid, subfolder="tokenizer", cache_dir=PRETRAINED_DIR,
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            mid, subfolder="tokenizer_2", cache_dir=PRETRAINED_DIR,
        )
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            mid, subfolder="text_encoder", cache_dir=PRETRAINED_DIR,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            mid, subfolder="text_encoder_2", cache_dir=PRETRAINED_DIR,
        )
        self.vae = AutoencoderKL.from_pretrained(mid, subfolder="vae", cache_dir=PRETRAINED_DIR)
        self.unet = UNet2DConditionModel.from_pretrained(
            mid, subfolder="unet", cache_dir=PRETRAINED_DIR,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            mid, subfolder="scheduler", cache_dir=PRETRAINED_DIR,
        )

        # // MVP Refactor: standard diffusers ControlNet (not from UNet copy — separate weights).
        self.controlnet = ControlNetModel.from_pretrained(
            model_cfg.controlnet_model_name_or_path,
            torch_dtype=weight_dtype,
            cache_dir=PRETRAINED_DIR,
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_cfg.clip_image_encoder_name_or_path,
            cache_dir=PRETRAINED_DIR,
        )
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            model_cfg.clip_image_encoder_name_or_path,
            cache_dir=PRETRAINED_DIR,
        )

        lora_config = LoraConfig(
            r=model_cfg.lora_rank,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            target_modules=list(model_cfg.lora_target_modules),
            init_lora_weights="gaussian",
        )
        self.unet = get_peft_model(self.unet, lora_config)

        clip_image_dim = self.image_encoder.config.projection_dim
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=model_cfg.ip_adapter_cross_attn_dim,
            clip_embeddings_dim=clip_image_dim,
            num_tokens=model_cfg.ip_adapter_num_tokens,
        )
        self.num_ip_tokens = model_cfg.ip_adapter_num_tokens
        self._install_ip_attn_processors()
        self._set_trainable_state()

    def _install_ip_attn_processors(self) -> None:
        attn_procs: Dict[str, Any] = {}
        base_unet = self.unet.base_model.model if hasattr(self.unet, "base_model") else self.unet
        cross_attention_dim = base_unet.config.cross_attention_dim

        for name in base_unet.attn_processors.keys():
            is_cross_attn = name.endswith("attn2.processor")
            if not is_cross_attn:
                from diffusers.models.attention_processor import AttnProcessor2_0
                attn_procs[name] = AttnProcessor2_0()
                continue

            parts = name.split(".")
            if name.startswith("mid_block"):
                hidden_size = base_unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(parts[1])
                hidden_size = list(reversed(base_unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(parts[1])
                hidden_size = base_unet.config.block_out_channels[block_id]
            else:
                raise ValueError(f"Unexpected attn processor location: {name}")

            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_tokens=self.num_ip_tokens,
            )
        base_unet.set_attn_processor(attn_procs)

    def _set_trainable_state(self) -> None:
        for m in (
            self.vae,
            self.text_encoder_one,
            self.text_encoder_two,
            self.image_encoder,
        ):
            m.requires_grad_(False)
            m.eval()

        for n, p in self.unet.named_parameters():
            p.requires_grad_("lora_" in n)

        base_unet = self.unet.base_model.model if hasattr(self.unet, "base_model") else self.unet
        for proc in base_unet.attn_processors.values():
            if isinstance(proc, IPAttnProcessor):
                for p in proc.parameters():
                    p.requires_grad_(True)

        self.controlnet.requires_grad_(True)
        self.image_proj_model.requires_grad_(True)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen = set()
        for p in list(self.unet.parameters()) + list(self.controlnet.parameters()) + list(
            self.image_proj_model.parameters()
        ):
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
        return params

    def to_device(self, device: Union[str, torch.device]) -> "CADSingleViewPipeline":
        self.device = torch.device(device)
        self.vae.to(device, dtype=self.weight_dtype)
        self.text_encoder_one.to(device, dtype=self.weight_dtype)
        self.text_encoder_two.to(device, dtype=self.weight_dtype)
        self.image_encoder.to(device, dtype=self.weight_dtype)
        self.unet.to(device)
        self.controlnet.to(device)
        self.image_proj_model.to(device)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds_list = []
        pooled_prompt_embeds: Optional[torch.Tensor] = None
        for tokenizer, text_encoder in zip(
            (self.tokenizer_one, self.tokenizer_two),
            (self.text_encoder_one, self.text_encoder_two),
        ):
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            outputs = text_encoder(input_ids, output_hidden_states=True)
            if isinstance(text_encoder, CLIPTextModelWithProjection):
                pooled_prompt_embeds = outputs[0]
            prompt_embeds_list.append(outputs.hidden_states[-2])
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        assert pooled_prompt_embeds is not None
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def encode_image_for_ip(self, images: Union[torch.Tensor, List[Image.Image], Image.Image]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            x = (images.float() + 1.0) / 2.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
            pixel_values = (x - mean) / std
        else:
            if isinstance(images, Image.Image):
                images = [images]
            pixel_values = self.clip_image_processor(
                images=images, return_tensors="pt"
            ).pixel_values.to(self.device)
        pixel_values = pixel_values.to(dtype=self.weight_dtype)
        return self.image_encoder(pixel_values).image_embeds

    def _build_added_cond_kwargs(
        self,
        batch_size: int,
        pooled_prompt_embeds: torch.Tensor,
        latent_shape: Tuple[int, ...],
    ) -> Dict[str, torch.Tensor]:
        h_latent, w_latent = latent_shape[-2], latent_shape[-1]
        h_pixel = h_latent * self.vae_scale_factor
        w_pixel = w_latent * self.vae_scale_factor
        add_time_ids = torch.tensor(
            [[h_pixel, w_pixel, 0, 0, h_pixel, w_pixel]],
            dtype=pooled_prompt_embeds.dtype,
            device=pooled_prompt_embeds.device,
        ).repeat(batch_size, 1)
        return {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def training_step_loss(
        self,
        i_final: torch.Tensor,
        condition_image: torch.Tensor,
        target_image: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """// MVP Refactor: MSE noise prediction; ``condition_image`` = prev-depth RGB."""
        b = target_image.shape[0]

        with torch.no_grad():
            latents = self.vae.encode(target_image.to(self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b,),
            device=latents.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompts)
        image_embeds = self.encode_image_for_ip(i_final)
        ip_tokens = self.image_proj_model(image_embeds.to(self.image_proj_model.proj.weight.dtype))
        encoder_hidden_states = torch.cat(
            [prompt_embeds, ip_tokens.to(prompt_embeds.dtype)], dim=1
        )

        added_cond_kwargs = self._build_added_cond_kwargs(
            batch_size=b,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latent_shape=latents.shape,
        )

        down_residuals, mid_residual = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=condition_image.to(self.weight_dtype),
            conditioning_scale=1.0,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": added_cond_kwargs["time_ids"],
            },
            return_dict=False,
        )

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=[d.to(noisy_latents.dtype) for d in down_residuals],
            mid_block_additional_residual=mid_residual.to(noisy_latents.dtype),
            return_dict=False,
        )[0]

        return F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    @torch.no_grad()
    def generate(
        self,
        i_final: Union[torch.Tensor, Image.Image],
        condition_image: torch.Tensor,
        prompt: str,
        negative_prompt: str = "blurry, distorted, low quality",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ) -> CADPipelineOutput:
        if condition_image.ndim == 3:
            condition_image = condition_image.unsqueeze(0)
        b = condition_image.shape[0]
        _, _, h_pix, w_pix = condition_image.shape

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        latent_shape = (
            b,
            self.unet.config.in_channels
            if hasattr(self.unet, "config")
            else self.unet.base_model.model.config.in_channels,
            h_pix // self.vae_scale_factor,
            w_pix // self.vae_scale_factor,
        )

        latents = torch.randn(
            latent_shape, generator=generator, device=self.device, dtype=self.weight_dtype
        )
        latents = latents * scheduler.init_noise_sigma

        pos_embeds, pos_pooled = self.encode_prompt([prompt] * b)
        neg_embeds, neg_pooled = self.encode_prompt([negative_prompt] * b)

        image_embeds = self.encode_image_for_ip(i_final)
        ip_tokens_pos = self.image_proj_model(
            image_embeds.to(self.image_proj_model.proj.weight.dtype)
        )
        ip_tokens_neg = torch.zeros_like(ip_tokens_pos)

        encoder_hidden_states_pos = torch.cat([pos_embeds, ip_tokens_pos.to(pos_embeds.dtype)], dim=1)
        encoder_hidden_states_neg = torch.cat([neg_embeds, ip_tokens_neg.to(neg_embeds.dtype)], dim=1)
        encoder_hidden_states = torch.cat([encoder_hidden_states_neg, encoder_hidden_states_pos], dim=0)
        pooled = torch.cat([neg_pooled, pos_pooled], dim=0)

        cn_cond = torch.cat([condition_image, condition_image], dim=0).to(self.weight_dtype)

        added_cond_kwargs = self._build_added_cond_kwargs(
            batch_size=2 * b,
            pooled_prompt_embeds=pooled,
            latent_shape=latents.shape,
        )

        for t in timesteps:
            latent_in = torch.cat([latents, latents], dim=0)
            latent_in = scheduler.scale_model_input(latent_in, t)

            down_residuals, mid_residual = self.controlnet(
                latent_in,
                t,
                encoder_hidden_states=torch.cat([neg_embeds, pos_embeds], dim=0),
                controlnet_cond=cn_cond,
                conditioning_scale=1.0,
                added_cond_kwargs={
                    "text_embeds": pooled,
                    "time_ids": added_cond_kwargs["time_ids"],
                },
                return_dict=False,
            )

            noise_pred = self.unet(
                latent_in,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=[d.to(latent_in.dtype) for d in down_residuals],
                mid_block_additional_residual=mid_residual.to(latent_in.dtype),
                return_dict=False,
            )[0]

            noise_pred_neg, noise_pred_pos = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents_to_decode = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents_to_decode.to(self.weight_dtype), return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        images_np = images.cpu().permute(0, 2, 3, 1).float().numpy()
        pil_images = [Image.fromarray((img * 255).round().astype("uint8")) for img in images_np]
        return CADPipelineOutput(images=pil_images, latents=latents)

    def save_trainables(
        self,
        path: str,
        unet_module: Optional[nn.Module] = None,
        controlnet_module: Optional[nn.Module] = None,
        image_proj_module: Optional[nn.Module] = None,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        unet = unet_module if unet_module is not None else self.unet
        cn = controlnet_module if controlnet_module is not None else self.controlnet
        iproj = image_proj_module if image_proj_module is not None else self.image_proj_model

        state: Dict[str, torch.Tensor] = {}
        for n, p in unet.named_parameters():
            if p.requires_grad:
                state[f"unet.{n}"] = p.detach().cpu()
        for n, p in cn.named_parameters():
            state[f"controlnet.{n}"] = p.detach().cpu()
        for n, p in iproj.named_parameters():
            state[f"image_proj.{n}"] = p.detach().cpu()
        torch.save(state, os.path.join(path, "trainables.pt"))

    def load_trainables(self, path: str, strict: bool = False) -> None:
        state: Dict[str, torch.Tensor] = torch.load(
            os.path.join(path, "trainables.pt"), map_location="cpu",
        )
        missing: List[str] = []
        for full, tensor in state.items():
            head, _, dot_name = full.partition(".")
            target = {
                "unet": self.unet,
                "controlnet": self.controlnet,
                "mv_controlnet": self.controlnet,  # // backward-compat old checkpoints
                "image_proj": self.image_proj_model,
            }.get(head)
            if target is None:
                missing.append(full)
                continue
            try:
                param = dict(target.named_parameters())[dot_name]
                with torch.no_grad():
                    param.copy_(tensor.to(param.device, dtype=param.dtype))
            except KeyError:
                missing.append(full)
        if missing and strict:
            raise RuntimeError(f"Missing keys: {missing[:5]} (total {len(missing)})")


# Backward-compatible alias
CADMultiViewPipeline = CADSingleViewPipeline
