"""Autoregressive multi-view step generator.

High-level flow::

    I_final  ──▶  [LLM Planner]  ──▶  [P_1, P_2, ..., P_n]            (prompts)
                                       │
                                       ▼
              G_prev = zeros           ┌─────────────┐
                  │                    │  Diffusion  │   for each P_k:
                  └──────────┬─────────▶│  Pipeline   │──▶ G_k = π(P_k, G_prev, I_final)
                             │         └─────────────┘     G_prev <- G_k
                             ▼                              save G_k to disk
                          (CLIP-Vision encoded once
                           as the global 3D reference)

Run::

    python inference.py \
        --i-final path/to/i_final.png \
        --checkpoint ./checkpoints/final \
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

from config import IFINAL_H, IFINAL_W, ModelConfig, NUM_ROWS, NUM_VIEWS, TILE_H, TILE_W
from models import CADMultiViewPipeline


# ===========================================================================
# 1. LLM Planner (Mock / Interface)
# ===========================================================================
class MLLMPlanner:
    """Abstract interface for the multimodal LLM step planner.

    Subclass and override :meth:`plan_steps`. The default implementation in
    :class:`MockMLLMPlanner` is intended only for tests / smoke runs.
    """

    def plan_steps(self, i_final: Image.Image) -> List[str]:
        raise NotImplementedError


class MockMLLMPlanner(MLLMPlanner):
    """Deterministic stub returning a small hard-coded plan.

    Useful for testing the whole pipeline end-to-end without a real MLLM.
    """

    def __init__(self, mock_plan: Optional[List[str]] = None) -> None:
        self.mock_plan = mock_plan or [
            "Sketch a 50x30mm rectangle on the XY plane.",
            "Extrude the rectangle by 20mm along +Z.",
            "Add a Ø10mm hole through the top face at the centre.",
            "Fillet all top edges with radius 2mm.",
        ]

    def plan_steps(self, i_final: Image.Image) -> List[str]:
        # Real implementation would feed `i_final` through Qwen2-VL:
        #   processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        #   model = Qwen2VLForConditionalGeneration.from_pretrained(...)
        #   messages = [{"role": "user", "content": [
        #       {"type": "image", "image": i_final},
        #       {"type": "text",  "text": "Decompose this CAD part into ..."},
        #   ]}]
        #   inputs = processor.apply_chat_template(messages, ...)
        #   plan_str = model.generate(...)
        #   return parse_plan(plan_str)
        _ = i_final
        return list(self.mock_plan)


# ---------------------------------------------------------------------------
def _save_grid_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save a ``(3, H, W)`` tensor in ``[-1, 1]`` as a PNG file."""
    x = (tensor.detach().float().cpu() + 1.0) / 2.0
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((x * 255).round().astype("uint8")).save(path)


# ===========================================================================
# 2. The autoregressive generator
# ===========================================================================
@dataclass
class GeneratorConfig:
    output_dir: str = "./generated"
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 0
    negative_prompt: str = "blurry, distorted, noisy, broken geometry"
    # If true, save G_prev (the input to each step) for debugging.
    save_inputs: bool = False


class CADAutoregressiveGenerator:
    """Drive the trained pipeline through the step plan.

    Parameters
    ----------
    pipeline:
        A fully constructed :class:`CADMultiViewPipeline`.
    planner:
        Object implementing :class:`MLLMPlanner`. Defaults to the mock planner.
    gen_cfg:
        Sampling hyperparameters and IO destination.
    """

    def __init__(
        self,
        pipeline: CADMultiViewPipeline,
        planner: Optional[MLLMPlanner] = None,
        gen_cfg: Optional[GeneratorConfig] = None,
    ) -> None:
        self.pipeline = pipeline
        self.planner = planner or MockMLLMPlanner()
        self.cfg = gen_cfg or GeneratorConfig()

        # Pre-build the I_final transform once.
        self._ifinal_tx = transforms.Compose([
            transforms.Resize((IFINAL_H, IFINAL_W), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    # ----------------------------------------------------------------- helpers
    def _load_i_final(self, i_final: Union[str, Image.Image]) -> Image.Image:
        if isinstance(i_final, str):
            return Image.open(i_final).convert("RGB")
        return i_final.convert("RGB") if i_final.mode != "RGB" else i_final

    def _zero_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initial G_prev for step k=0."""
        return torch.zeros(
            (1, 3, NUM_ROWS * TILE_H, NUM_VIEWS * TILE_W),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _to_grid_tensor(image: Image.Image) -> torch.Tensor:
        """Convert a PIL grid image back into a normalized tensor for the next step."""
        tx = transforms.Compose([
            transforms.Resize((NUM_ROWS * TILE_H, NUM_VIEWS * TILE_W), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return tx(image).unsqueeze(0)  # (1, 3, NUM_ROWS*H, NUM_VIEWS*W)

    # ----------------------------------------------------------------- main entry
    def run(
        self,
        i_final: Union[str, Image.Image],
        run_name: str = "run",
    ) -> List[Image.Image]:
        """Execute the full autoregressive loop and return the generated grids."""
        out_dir = os.path.join(self.cfg.output_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)

        # Step 1: ask the MLLM planner for the step list.
        i_final_pil = self._load_i_final(i_final)
        prompts = self.planner.plan_steps(i_final_pil)
        with open(os.path.join(out_dir, "plan.txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(f"[{i}] {p}" for i, p in enumerate(prompts)))

        # Pre-encode I_final into a tensor used both as IP-Adapter input and
        # for logging. The pipeline can also accept a PIL image directly.
        device = self.pipeline.device
        dtype = self.pipeline.weight_dtype

        i_final_tensor = self._ifinal_tx(i_final_pil).unsqueeze(0).to(device, dtype=dtype)

        # Step 2 & 3: autoregressive loop.
        generated_grids: List[Image.Image] = []
        g_prev = self._zero_grid(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(self.cfg.seed)

        for k, p_k in enumerate(prompts):
            if self.cfg.save_inputs:
                _save_grid_tensor(g_prev[0], os.path.join(out_dir, f"step_{k:03d}_input.png"))

            output = self.pipeline.generate(
                i_final=i_final_tensor,
                g_prev=g_prev.to(dtype=dtype),
                prompt=p_k,
                negative_prompt=self.cfg.negative_prompt,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                generator=generator,
            )
            g_k_image: Image.Image = output.images[0]
            g_k_image.save(os.path.join(out_dir, f"step_{k:03d}.png"))
            generated_grids.append(g_k_image)

            # Update G_prev for the next iteration. We re-normalize the PIL
            # output back to [-1, 1] so the ControlNet sees the same data
            # distribution as during training.
            g_prev = self._to_grid_tensor(g_k_image).to(device, dtype=torch.float32)

        return generated_grids


# ===========================================================================
# CLI entry point
# ===========================================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--i-final", type=str, required=True, help="Path to a single CAD reference view PNG.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Folder produced by train.py (containing trainables.pt).")
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
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_cfg = ModelConfig()
    pipeline = CADMultiViewPipeline(
        model_cfg=model_cfg,
        device=device,
        weight_dtype=dtype,
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
    generator = CADAutoregressiveGenerator(
        pipeline=pipeline,
        planner=MockMLLMPlanner(),   # plug in a real Qwen2-VL planner here
        gen_cfg=gen_cfg,
    )

    grids = generator.run(i_final=args.i_final, run_name=args.run_name)
    print(f"Generated {len(grids)} step grids under {os.path.join(args.output_dir, args.run_name)}")


if __name__ == "__main__":
    main()
