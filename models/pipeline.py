"""End-to-end Diffusion pipeline used by both training and inference.

Components:
    * Frozen SDXL VAE                (encodes G_target -> latents, decodes back)
    * Frozen SDXL text encoder(s)   (encodes the per-step prompt)
    * SDXL UNet with LoRA adapters  (trainable LoRA only)
    * MultiViewControlNetModel      (trainable, hacks `controlnet_cond_embedding`)
    * Frozen CLIP-Vision encoder    (extracts global features of I_final)
    * ImageProjModel (IP-Adapter)   (trainable, projects CLIP feats -> K tokens)
    * IPAttnProcessor               (parallel K/V branch in every cross-attn)

Trainable parameters (during fine-tuning):
    1. LoRA injected into UNet.
    2. The MV-ControlNet (entire model).
    3. ``ImageProjModel`` + the ``to_k_ip / to_v_ip`` linears inside every
       :class:`IPAttnProcessor` (i.e. the IP-Adapter projection layers).

Everything else is frozen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
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

from config import ModelConfig, NUM_ROWS, NUM_VIEWS, PRETRAINED_DIR, TILE_H, TILE_W
from .mv_controlnet import MultiViewControlNetModel


# ===========================================================================
# IP-Adapter components
# ===========================================================================
class ImageProjModel(nn.Module):
    """Project a CLIP image embedding to ``num_tokens`` cross-attn tokens.

    The official IP-Adapter (Tencent) uses exactly this design:

        x: (B, image_embed_dim)
        -> Linear -> (B, num_tokens * cross_attn_dim)
        -> reshape -> (B, num_tokens, cross_attn_dim)
        -> LayerNorm
    """

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
        # (B, clip_embeddings_dim) -> (B, num_tokens, cross_attention_dim)
        b = image_embeds.shape[0]
        x = self.proj(image_embeds).reshape(b, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


class IPAttnProcessor(nn.Module):
    """Cross-attention processor adding a parallel K/V branch for IP tokens.

    The original ``encoder_hidden_states`` (text) and the additional
    ``ip_hidden_states`` (image tokens from :class:`ImageProjModel`) are
    *concatenated along the sequence axis*; that is mathematically the same
    as running two attentions and summing their outputs, but cheaper.

    Only ``to_k_ip`` and ``to_v_ip`` are trainable -- ``to_q`` stays the
    UNet's pretrained query projection (LoRA may modify it).
    """

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
        # Parallel K/V for the image-token branch.
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
        # `encoder_hidden_states` is augmented by the pipeline so that the
        # *first* `seq_text` tokens are the text features and the *last*
        # `num_tokens` tokens are the IP image features.
        residual = hidden_states

        # Spatial flatten if needed (UNet sometimes feeds 4D tensors).
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        b, _, _ = hidden_states.shape
        if encoder_hidden_states is None:
            # Self-attention path (no IP injection here).
            encoder_hidden_states = hidden_states
            text_kv, ip_kv = encoder_hidden_states, None
        else:
            # Split the concatenated text + IP sequence.
            text_kv = encoder_hidden_states[:, : -self.num_tokens, :]
            ip_kv = encoder_hidden_states[:, -self.num_tokens :, :]

        # ---- Standard cross-attention with the text branch ----
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        key   = attn.to_k(text_kv)
        value = attn.to_v(text_kv)

        head_dim = query.shape[-1] // attn.heads
        query = query.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(b, -1, attn.heads, head_dim).transpose(1, 2)

        text_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # ---- Parallel IP branch ----
        if ip_kv is not None:
            ip_key   = self.to_k_ip(ip_kv)
            ip_value = self.to_v_ip(ip_kv)
            ip_key   = ip_key.view(b, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(b, -1, attn.heads, head_dim).transpose(1, 2)
            ip_out = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            out = text_out + self.scale * ip_out
        else:
            out = text_out

        # Merge heads and project.
        out = out.transpose(1, 2).reshape(b, -1, attn.heads * head_dim)
        out = out.to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)  # dropout

        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(b, c, h, w)

        if attn.residual_connection:
            out = out + residual

        return out / attn.rescale_output_factor


# ===========================================================================
# The pipeline itself
# ===========================================================================
@dataclass
class CADPipelineOutput:
    """Container returned by :meth:`CADMultiViewPipeline.generate`."""

    images: List[Image.Image]
    latents: torch.Tensor          # (B, 4, H_lat, W_lat)


class CADMultiViewPipeline(nn.Module):
    """Custom SDXL pipeline for autoregressive multi-view step generation.

    This class wraps the seven sub-models and exposes:
        * :meth:`encode_prompt`            (text -> SDXL prompt embeds + pooled)
        * :meth:`encode_image_for_ip`     (PIL/Tensor -> CLIP image embeds)
        * :meth:`training_step_loss`     (MSE noise loss used by train.py)
        * :meth:`generate`                  (CFG denoising used by inference.py)
        * :meth:`get_trainable_parameters`  (helper for the optimizer)

    Note:
        We deliberately *do not* inherit from ``DiffusionPipeline``. The
        ``save_pretrained / from_pretrained`` infrastructure makes too many
        assumptions about our novel components (e.g. multi-view ControlNet);
        we ship our own minimal save/load below.
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

        # ---- SDXL base components -----------------------------------------
        # Every ``from_pretrained`` call routes through ``PRETRAINED_DIR`` so
        # weights land inside the project's ``pretrained_lm/`` folder rather
        # than ``~/.cache/huggingface``.
        # Tokenizers (two for SDXL) + text encoders.
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            model_cfg.pretrained_model_name_or_path,
            subfolder="tokenizer",
            cache_dir=PRETRAINED_DIR,
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            model_cfg.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            cache_dir=PRETRAINED_DIR,
        )
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            model_cfg.pretrained_model_name_or_path,
            subfolder="text_encoder",
            cache_dir=PRETRAINED_DIR,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_cfg.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            cache_dir=PRETRAINED_DIR,
        )

        self.vae = AutoencoderKL.from_pretrained(
            model_cfg.pretrained_model_name_or_path,
            subfolder="vae",
            cache_dir=PRETRAINED_DIR,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_cfg.pretrained_model_name_or_path,
            subfolder="unet",
            cache_dir=PRETRAINED_DIR,
        )
        # Noise scheduler used for training. Inference can swap this for a
        # DPM-Solver / Euler variant if desired.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            cache_dir=PRETRAINED_DIR,
        )

        # ---- CLIP-Vision for IP-Adapter ------------------------------------
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_cfg.clip_image_encoder_name_or_path,
            cache_dir=PRETRAINED_DIR,
        )
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            model_cfg.clip_image_encoder_name_or_path,
            cache_dir=PRETRAINED_DIR,
        )

        # ---- LoRA injection into UNet (PEFT) -------------------------------
        lora_config = LoraConfig(
            r=model_cfg.lora_rank,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            target_modules=list(model_cfg.lora_target_modules),
            init_lora_weights="gaussian",
        )
        # ``get_peft_model`` wraps the UNet so calls to .forward still work
        # but with LoRA deltas applied to the matched submodules.
        self.unet = get_peft_model(self.unet, lora_config)

        # ---- Multi-View ControlNet (init from UNet for compatible block shapes) --
        # NOTE: ``diffusers.ControlNetModel.from_unet`` only accepts
        # ``controlnet_conditioning_channel_order``, ``load_weights_from_unet``,
        # and ``conditioning_channels``. Passing ``conditioning_embedding_out_channels``
        # to it raises TypeError on diffusers >= 0.27. We therefore build a
        # stock ControlNet first and replace its conditioning embedding via
        # ``install_multiview_embedding`` -- that's where our 16/32/96/256
        # block sizes (or whatever ``mv_cn_block_out_channels`` specifies)
        # actually take effect.
        base_unet = self.unet.base_model.model if hasattr(self.unet, "base_model") else self.unet
        self.mv_controlnet = MultiViewControlNetModel.from_unet(
            base_unet,
            conditioning_channels=3,
        )
        self.mv_controlnet.install_multiview_embedding(
            num_heads=model_cfg.mv_cn_num_attn_heads,
            block_out_channels=model_cfg.mv_cn_block_out_channels,
        )

        # ---- IP-Adapter image projection + attention processors -----------
        # Find the CLIP image-projection output dim.
        clip_image_dim = self.image_encoder.config.projection_dim
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=model_cfg.ip_adapter_cross_attn_dim,
            clip_embeddings_dim=clip_image_dim,
            num_tokens=model_cfg.ip_adapter_num_tokens,
        )
        self.num_ip_tokens = model_cfg.ip_adapter_num_tokens
        self._install_ip_attn_processors()

        # ---- Freeze everything except the three trainable groups ---------
        self._set_trainable_state()

    # ----------------------------------------------------------------- setup
    def _install_ip_attn_processors(self) -> None:
        """Replace every cross-attention processor in the UNet with IPAttnProcessor.

        Self-attention layers keep the default processor (``attn1`` in
        diffusers naming); cross-attention layers (``attn2``) get the
        IP-augmented one.
        """
        attn_procs: Dict[str, Any] = {}
        # PEFT wraps the UNet -- reach the inner model for ``attn_processors``.
        base_unet = self.unet.base_model.model if hasattr(self.unet, "base_model") else self.unet
        cross_attention_dim = base_unet.config.cross_attention_dim

        for name in base_unet.attn_processors.keys():
            # `name` ends with "...attn1.processor" or "...attn2.processor".
            is_cross_attn = name.endswith("attn2.processor")
            if not is_cross_attn:
                # Keep default self-attention processor.
                from diffusers.models.attention_processor import AttnProcessor2_0
                attn_procs[name] = AttnProcessor2_0()
                continue

            # Figure out `hidden_size` from the module hierarchy. Attention
            # processor names look like ``<scope>.<block_id>.<...>.processor``
            # for down/up blocks, or ``mid_block.<...>.processor``. We parse
            # the integer block id robustly by splitting on '.'  (avoids the
            # 1-char slice trick which silently mis-parses block ids >= 10).
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
        """Freeze all base components; mark only LoRA / MV-CN / IP-Adapter trainable."""
        # Freeze everything by default.
        for m in (
            self.vae,
            self.text_encoder_one,
            self.text_encoder_two,
            self.image_encoder,
        ):
            m.requires_grad_(False)
            m.eval()

        # Freeze the UNet *base* (PEFT keeps LoRA params trainable automatically).
        for n, p in self.unet.named_parameters():
            # PEFT names LoRA params with ".lora_" in them.
            p.requires_grad_("lora_" in n)

        # The IP-Adapter attention processors *inside* the UNet were just
        # created above and PEFT did *not* tag them with "lora_", so we need
        # to explicitly enable grads for the to_k_ip / to_v_ip linears.
        base_unet = self.unet.base_model.model if hasattr(self.unet, "base_model") else self.unet
        for proc in base_unet.attn_processors.values():
            if isinstance(proc, IPAttnProcessor):
                for p in proc.parameters():
                    p.requires_grad_(True)

        # MV-ControlNet is fully trainable.
        self.mv_controlnet.requires_grad_(True)

        # Image projector is fully trainable.
        self.image_proj_model.requires_grad_(True)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return a deduplicated list of params for the optimizer."""
        params: List[nn.Parameter] = []
        seen = set()
        for p in list(self.unet.parameters()) \
                + list(self.mv_controlnet.parameters()) \
                + list(self.image_proj_model.parameters()):
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
        return params

    # ----------------------------------------------------------------- to(device)
    def to_device(self, device: Union[str, torch.device]) -> "CADMultiViewPipeline":
        """Move all sub-models to ``device``; keep the chosen dtype on frozen parts."""
        self.device = torch.device(device)
        # Frozen models can live in lower precision.
        self.vae.to(device, dtype=self.weight_dtype)
        self.text_encoder_one.to(device, dtype=self.weight_dtype)
        self.text_encoder_two.to(device, dtype=self.weight_dtype)
        self.image_encoder.to(device, dtype=self.weight_dtype)
        # Trainable models stay in fp32 by default; cast manually if needed.
        self.unet.to(device)
        self.mv_controlnet.to(device)
        self.image_proj_model.to(device)
        return self

    # ----------------------------------------------------------------- encoders
    @torch.no_grad()
    def encode_prompt(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SDXL prompt encoding: returns (prompt_embeds, pooled_prompt_embeds)."""
        prompt_embeds_list = []
        pooled_prompt_embeds: Optional[torch.Tensor] = None
        tokenizers = [self.tokenizer_one, self.tokenizer_two]
        text_encoders = [self.text_encoder_one, self.text_encoder_two]

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            outputs = text_encoder(input_ids, output_hidden_states=True)
            # Second tokenizer's text encoder yields the pooled embedding
            # SDXL conditions on.
            if isinstance(text_encoder, CLIPTextModelWithProjection):
                pooled_prompt_embeds = outputs[0]
            # Penultimate hidden state (SDXL convention).
            prompt_embeds_list.append(outputs.hidden_states[-2])

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        assert pooled_prompt_embeds is not None
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def encode_image_for_ip(
        self,
        images: Union[torch.Tensor, List[Image.Image], Image.Image],
    ) -> torch.Tensor:
        """Pass ``I_final`` through CLIP-Vision, return image embeddings.

        ``images`` may be:
            * a ``(B, 3, H, W)`` tensor in [-1, 1]   (training path)
            * a single PIL.Image or list of PIL.Images (inference path)
        """
        if isinstance(images, torch.Tensor):
            # Tensor in [-1, 1]; the CLIPImageProcessor expects PIL or
            # numpy with proper normalization. We rebuild manually:
            #   (x + 1) / 2 -> [0, 1] -> CLIP mean/std.
            x = (images.float() + 1.0) / 2.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
            pixel_values = (x - mean) / std
        else:
            if isinstance(images, Image.Image):
                images = [images]
            pixel_values = self.clip_image_processor(
                images=images, return_tensors="pt"
            ).pixel_values.to(self.device)

        pixel_values = pixel_values.to(dtype=self.weight_dtype)
        out = self.image_encoder(pixel_values)
        # `image_embeds` (B, projection_dim) – the global pooled CLIP feature.
        return out.image_embeds

    # ----------------------------------------------------------------- helpers
    def _build_added_cond_kwargs(
        self,
        batch_size: int,
        pooled_prompt_embeds: torch.Tensor,
        latent_shape: Tuple[int, ...],
    ) -> Dict[str, torch.Tensor]:
        """SDXL needs ``time_ids`` and a pooled text embed at every step."""
        # We use the native latent resolution as both "original" and "target"
        # resolutions, with zero crop offsets. SDXL was trained with such
        # synthetic micro-conditioning when no metadata is available.
        h_latent, w_latent = latent_shape[-2], latent_shape[-1]
        h_pixel = h_latent * self.vae_scale_factor
        w_pixel = w_latent * self.vae_scale_factor
        add_time_ids = torch.tensor(
            [[h_pixel, w_pixel, 0, 0, h_pixel, w_pixel]],
            dtype=pooled_prompt_embeds.dtype,
            device=pooled_prompt_embeds.device,
        ).repeat(batch_size, 1)
        return {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    # ----------------------------------------------------------------- training
    def training_step_loss(
        self,
        i_final: torch.Tensor,
        g_prev: torch.Tensor,
        g_target: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """One forward pass of the noise-prediction MSE loss.

        Shapes:
            i_final  : (B, 3, IFINAL_H, IFINAL_W)
            g_prev   : (B, 3, NUM_ROWS*TILE_H, NUM_VIEWS*TILE_W)
            g_target : same as g_prev
            prompts  : List[str] of length B
        """
        b = g_target.shape[0]

        # 1) Encode G_target into latents.
        with torch.no_grad():
            latents = self.vae.encode(g_target.to(self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # 2) Add noise.
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (b,), device=latents.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 3) Encode text and CLIP-Vision conditioning.
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompts)
        image_embeds = self.encode_image_for_ip(i_final)
        ip_tokens = self.image_proj_model(image_embeds.to(self.image_proj_model.proj.weight.dtype))
        # Concatenate text tokens with IP image tokens along the sequence axis.
        encoder_hidden_states = torch.cat(
            [prompt_embeds, ip_tokens.to(prompt_embeds.dtype)], dim=1
        )

        # 4) Multi-View ControlNet residuals.
        # ControlNet expects the control image at the *latent* resolution? No:
        # diffusers' ControlNet handles arbitrary input via its
        # `controlnet_cond_embedding` (which is exactly what we replaced).
        # We feed the raw 4x8 tiled grid.
        added_cond_kwargs = self._build_added_cond_kwargs(
            batch_size=b,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latent_shape=latents.shape,
        )
        # Diffusers' ControlNet forward returns (down_block_res_samples,
        # mid_block_res_sample).
        down_residuals, mid_residual = self.mv_controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,   # text-only for ControlNet
            controlnet_cond=g_prev.to(self.weight_dtype),
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": added_cond_kwargs["time_ids"],
            },
            return_dict=False,
        )

        # 5) UNet prediction.
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=[d.to(noisy_latents.dtype) for d in down_residuals],
            mid_block_additional_residual=mid_residual.to(noisy_latents.dtype),
            return_dict=False,
        )[0]

        # 6) Standard epsilon-prediction MSE.
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        return loss

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(
        self,
        i_final: Union[torch.Tensor, Image.Image],
        g_prev:  torch.Tensor,
        prompt:  str,
        negative_prompt: str = "blurry, distorted, low quality",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ) -> CADPipelineOutput:
        """Sample one modeling-step grid given previous grid + reference image."""
        # Make sure the scheduler is in inference mode.
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        # Pull a batch dim out of g_prev if needed.
        if g_prev.ndim == 3:
            g_prev = g_prev.unsqueeze(0)
        b = g_prev.shape[0]

        # Compute the latent spatial size from the desired output size.
        h_pix = NUM_ROWS * TILE_H
        w_pix = NUM_VIEWS * TILE_W
        latent_shape = (
            b,
            self.unet.config.in_channels if hasattr(self.unet, "config")
            else (self.unet.base_model.model.config.in_channels),
            h_pix // self.vae_scale_factor,
            w_pix // self.vae_scale_factor,
        )

        latents = torch.randn(latent_shape, generator=generator, device=self.device,
                              dtype=self.weight_dtype)
        latents = latents * scheduler.init_noise_sigma

        # Encode the text prompts (positive + negative) for CFG.
        pos_embeds, pos_pooled = self.encode_prompt([prompt] * b)
        neg_embeds, neg_pooled = self.encode_prompt([negative_prompt] * b)

        # CLIP-Vision -> IP tokens for both branches:
        # for the negative branch we use a zero image embedding so the
        # adapter only fires for the positive branch.
        image_embeds = self.encode_image_for_ip(i_final)
        ip_tokens_pos = self.image_proj_model(
            image_embeds.to(self.image_proj_model.proj.weight.dtype)
        )
        ip_tokens_neg = torch.zeros_like(ip_tokens_pos)

        encoder_hidden_states_pos = torch.cat([pos_embeds, ip_tokens_pos.to(pos_embeds.dtype)], dim=1)
        encoder_hidden_states_neg = torch.cat([neg_embeds, ip_tokens_neg.to(neg_embeds.dtype)], dim=1)

        # Concatenate negative/positive batches to do a single UNet call.
        encoder_hidden_states = torch.cat([encoder_hidden_states_neg, encoder_hidden_states_pos], dim=0)
        pooled = torch.cat([neg_pooled, pos_pooled], dim=0)

        # ControlNet conditioning (same for both halves).
        cn_cond = torch.cat([g_prev, g_prev], dim=0).to(self.weight_dtype)

        added_cond_kwargs = self._build_added_cond_kwargs(
            batch_size=2 * b,
            pooled_prompt_embeds=pooled,
            latent_shape=latents.shape,
        )

        for t in timesteps:
            latent_in = torch.cat([latents, latents], dim=0)
            latent_in = scheduler.scale_model_input(latent_in, t)

            down_residuals, mid_residual = self.mv_controlnet(
                latent_in,
                t,
                encoder_hidden_states=torch.cat([neg_embeds, pos_embeds], dim=0),
                controlnet_cond=cn_cond,
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

        # Decode.
        latents_to_decode = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents_to_decode.to(self.weight_dtype), return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        pil_images = [Image.fromarray((img * 255).round().astype("uint8")) for img in images]
        return CADPipelineOutput(images=pil_images, latents=latents)

    # ----------------------------------------------------------------- (de)serialization
    def save_trainables(
        self,
        path: str,
        unet_module: Optional[nn.Module] = None,
        mv_controlnet_module: Optional[nn.Module] = None,
        image_proj_module: Optional[nn.Module] = None,
    ) -> None:
        """Save just the trainable params (LoRA + MV-CN + IP-Adapter).

        Under ``accelerate``/DDP, ``pipeline.unet`` (etc.) are wrappers whose
        ``named_parameters()`` keys carry a ``module.`` prefix. Pass the
        unwrapped modules via the optional kwargs to strip that prefix --
        otherwise the saved keys won't match :meth:`load_trainables`.

        Typical caller in ``train.py``::

            pipeline.save_trainables(
                ckpt_dir,
                unet_module=accelerator.unwrap_model(pipeline.unet),
                mv_controlnet_module=accelerator.unwrap_model(pipeline.mv_controlnet),
                image_proj_module=accelerator.unwrap_model(pipeline.image_proj_model),
            )
        """
        os.makedirs(path, exist_ok=True)

        unet  = unet_module          if unet_module          is not None else self.unet
        mvcn  = mv_controlnet_module if mv_controlnet_module is not None else self.mv_controlnet
        iproj = image_proj_module    if image_proj_module    is not None else self.image_proj_model

        state: Dict[str, torch.Tensor] = {}
        # UNet: LoRA params + IP-Adapter to_k_ip/to_v_ip (anything trainable).
        for n, p in unet.named_parameters():
            if p.requires_grad:
                state[f"unet.{n}"] = p.detach().cpu()
        # MV-ControlNet (fully trainable).
        for n, p in mvcn.named_parameters():
            state[f"mv_controlnet.{n}"] = p.detach().cpu()
        # ImageProj (fully trainable).
        for n, p in iproj.named_parameters():
            state[f"image_proj.{n}"] = p.detach().cpu()
        torch.save(state, os.path.join(path, "trainables.pt"))

    def load_trainables(self, path: str, strict: bool = False) -> None:
        """Inverse of :meth:`save_trainables`.

        Loads into ``self.{unet, mv_controlnet, image_proj_model}`` -- the
        UN-wrapped (pre-accelerate) modules. Call BEFORE ``accelerator.prepare``.
        """
        state: Dict[str, torch.Tensor] = torch.load(
            os.path.join(path, "trainables.pt"), map_location="cpu",
        )
        missing: List[str] = []
        for full, tensor in state.items():
            head, _, dot_name = full.partition(".")
            target = {
                "unet": self.unet,
                "mv_controlnet": self.mv_controlnet,
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
