"""Autoregressive single-view step generator (MVP).

// MVP Refactor: one canonical view (``config.MVP_VIEW_SUFFIX``) and training
resolution — no multi-tile ``G_prev`` grid.

High-level flow::

    I_final  ──▶  [LLM Planner]  ──▶  [P_1, P_2, ..., P_n]
                                       │
                                       ▼
       condition = zeros   ┌─────────────────────┐
            │              │  SDXL + ControlNet   │
            └──────┬──────▶│  + IP-Adapter       │──▶ image_k for each P_k
                   │       └─────────────────────┘
                   │              condition <- heuristic(prev image)
                   ▼
            (CLIP-Vision encodes I_final once)

Training used ``prev_depth_map.png`` as ControlNet input; at inference we do not
have GT depth, so step 0 uses a zero map and later steps reuse a grayscale
encoding of the previous RGB output (MVP heuristic — swap for a depth
estimator when available).

Run::

    python inference.py \\
        --i-final path/to/final_snapshot.png \\
        --checkpoint ./checkpoints/final \\
        --output-dir ./generated/part_demo
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import IFINAL_H, IFINAL_W, ModelConfig, TRAIN_IMAGE_H, TRAIN_IMAGE_W
from models import CADSingleViewPipeline


# ===========================================================================
# 1. LLM Planner (Mock / Interface)
# ===========================================================================
class MLLMPlanner:
    """Abstract interface for the multimodal LLM step planner."""

    def plan_steps(self, i_final: Image.Image) -> List[str]:
        raise NotImplementedError


class MockMLLMPlanner(MLLMPlanner):
    """Deterministic stub returning a small hard-coded plan."""

    def __init__(self, mock_plan: Optional[List[str]] = None) -> None:
        self.mock_plan = mock_plan or [
            "Sketch a 50x30mm rectangle on the XY plane.",
            "Extrude the rectangle by 20mm along +Z.",
            "Add a Ø10mm hole through the top face at the centre.",
            "Fillet all top edges with radius 2mm.",
        ]

    def plan_steps(self, i_final: Image.Image) -> List[str]:
        _ = i_final
        return list(self.mock_plan)


def _save_image_tensor(tensor_chw: torch.Tensor, path: str) -> None:
    """Save ``(3, H, W)`` in ``[-1, 1]`` as PNG."""
    x = (tensor_chw.detach().float().cpu() + 1.0) / 2.0
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((x * 255).round().astype("uint8")).save(path)


def _zero_condition(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch, 3, TRAIN_IMAGE_H, TRAIN_IMAGE_W, device=device, dtype=dtype)


def _rgb_pil_to_depth_style_condition(
    pil: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """MVP: mimic dataset depth conditioning — grayscale replicated to 3 ch, [-1, 1]."""
    img = pil.convert("L").resize((TRAIN_IMAGE_W, TRAIN_IMAGE_H), Image.Resampling.BILINEAR)
    t = TF.to_tensor(img).to(device=device, dtype=torch.float32)
    t = t.expand(3, -1, -1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(dtype)


# ===========================================================================
# 2. Autoregressive generator
# ===========================================================================
@dataclass
class GeneratorConfig:
    output_dir: str = "./generated"
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 0
    negative_prompt: str = "blurry, distorted, noisy, broken geometry"
    save_inputs: bool = False


class CADAutoregressiveGenerator:
    """Drive :class:`CADSingleViewPipeline` with a per-step prompt plan."""

    def __init__(
        self,
        pipeline: CADSingleViewPipeline,
        planner: Optional[MLLMPlanner] = None,
        gen_cfg: Optional[GeneratorConfig] = None,
    ) -> None:
        self.pipeline = pipeline
        self.planner = planner or MockMLLMPlanner()
        self.cfg = gen_cfg or GeneratorConfig()
        self._ifinal_tx = transforms.Compose([
            transforms.Resize((IFINAL_H, IFINAL_W), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _load_i_final(self, i_final: Union[str, Image.Image]) -> Image.Image:
        if isinstance(i_final, str):
            return Image.open(i_final).convert("RGB")
        return i_final.convert("RGB") if i_final.mode != "RGB" else i_final

    def run(
        self,
        i_final: Union[str, Image.Image],
        run_name: str = "run",
    ) -> List[Image.Image]:
        out_dir = os.path.join(self.cfg.output_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)

        i_final_pil = self._load_i_final(i_final)
        prompts = self.planner.plan_steps(i_final_pil)
        with open(os.path.join(out_dir, "plan.txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(f"[{i}] {p}" for i, p in enumerate(prompts)))

        device = self.pipeline.device
        dtype = self.pipeline.weight_dtype
        i_final_tensor = self._ifinal_tx(i_final_pil).unsqueeze(0).to(device, dtype=dtype)

        generated: List[Image.Image] = []
        generator = torch.Generator(device=device).manual_seed(self.cfg.seed)

        for k, p_k in enumerate(prompts):
            if k == 0:
                cond = _zero_condition(1, device=device, dtype=dtype)
            else:
                cond = _rgb_pil_to_depth_style_condition(generated[-1], device=device, dtype=dtype)

            if self.cfg.save_inputs:
                _save_image_tensor(cond[0], os.path.join(out_dir, f"step_{k:03d}_condition.png"))

            output = self.pipeline.generate(
                i_final=i_final_tensor,
                condition_image=cond,
                prompt=p_k,
                negative_prompt=self.cfg.negative_prompt,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                generator=generator,
            )
            img: Image.Image = output.images[0]
            img.save(os.path.join(out_dir, f"step_{k:03d}.png"))
            generated.append(img)

        return generated


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--i-final", type=str, required=True, help="Path to final_snapshot-style PNG.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Folder from train.py (trainables.pt).",
    )
    p.add_argument("--output-dir", type=str, default="./generated")
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-inputs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_dtype = torch.float16 if device == "cuda" else torch.float32

    model_cfg = ModelConfig()
    pipeline = CADSingleViewPipeline(
        model_cfg=model_cfg,
        device=device,
        weight_dtype=infer_dtype,
    ).to_device(device)

    if args.checkpoint is not None:
        pipeline.load_trainables(args.checkpoint, strict=False)

    gen_cfg = GeneratorConfig(
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        save_inputs=args.save_inputs,
    )
    generator = CADAutoregressiveGenerator(pipeline=pipeline, gen_cfg=gen_cfg)
    images = generator.run(i_final=args.i_final, run_name=args.run_name)
    print(f"Generated {len(images)} step images under {os.path.join(args.output_dir, args.run_name)}")


if __name__ == "__main__":
    main()
