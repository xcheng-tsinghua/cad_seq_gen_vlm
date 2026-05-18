"""Phase 4 — Autoregressive single-view reverse modeling loop (MVP).

Semantic–geometric split:

* **Planner** (Qwen after Phase 3, or mock here): ``(I_final, state_k) → Prompt_{k+1}``.
* **Renderer** (SDXL + ControlNet + IP-Adapter after Phase 2): ``(I_final, depth_k, prompt) → G_{k+1}``
  (painter-style ``overlayed_all``).

**Initialization:** ``final_snapshot`` + base depth ``depth_0`` (``main_ref_depth.png`` or explicit path).
Each iteration:

1. **Plan** — planner proposes the next painter instruction (no ``operation_param.json`` at inference).
2. **Render** — diffusion predicts the overlay composite conditioned on ``depth_k``.
3. **Execute** — CAD boolean + re-render of ``depth_{k+1}`` is **out of scope**; this script chains steps by
   reusing the generated overlay RGB as the planner ``state_{k+1}`` and approximates the next ControlNet map
   from overlay luminance (closing the loop requires your CAD kernel).

Stop when the planner returns :data:`config.PLANNER_STOP_TOKEN` or ``max_steps`` is hit.

// MVP Refactor: no multi-tile grids; standard ``CADSingleViewPipeline`` (diffusers ControlNet + IP-Adapter).

Run::

    python inference.py \\
        --i-final path/to/final_snapshot.png \\
        --depth-0 path/to/main_ref_depth.png \\
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

from config import IFINAL_H, IFINAL_W, ModelConfig, PLANNER_STOP_TOKEN, TRAIN_IMAGE_H, TRAIN_IMAGE_W
from models import CADSingleViewPipeline


# -----------------------------------------------------------------------------
# Phase 4 — Planner interface (Phase 3 checkpoint would plug in here)
# -----------------------------------------------------------------------------
class MLLMPlanner:
    def predict_next(self, i_final: Image.Image, state: Image.Image) -> str:
        raise NotImplementedError


class MockMLLMPlanner(MLLMPlanner):
    """Deterministic stub until a fine-tuned Qwen is loaded."""

    def __init__(self, mock_steps: Optional[List[str]] = None) -> None:
        self._steps = list(mock_steps or [
            "Sketch a rectangle on the base plane using red guide curves, add yellow sketch-plane tint.",
            "Extrude that profile into a green boss capped by a blue termination face toward the front mass.",
            "Cut a magenta pocket from the top using the red loop on the upper face.",
        ])

    def predict_next(self, i_final: Image.Image, state: Image.Image) -> str:
        del i_final, state
        if not self._steps:
            return PLANNER_STOP_TOKEN
        return self._steps.pop(0)


def _save_image_tensor(tensor_chw: torch.Tensor, path: str) -> None:
    x = (tensor_chw.detach().float().cpu() + 1.0) / 2.0
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((x * 255).round().astype("uint8")).save(path)


def _zero_condition(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch, 3, TRAIN_IMAGE_H, TRAIN_IMAGE_W, device=device, dtype=dtype)


def _load_depth_png_as_rgb_pil(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L").convert("RGB") if img.mode == "L" else img.convert("RGB")


def _pil_to_controlnet_tensor(pil: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Depth-style map: luminance → 3× channel, normalized to [-1, 1] at train resolution."""
    gray = pil.convert("L").resize((TRAIN_IMAGE_W, TRAIN_IMAGE_H), Image.Resampling.BILINEAR)
    t = TF.to_tensor(gray).to(device=device, dtype=torch.float32)
    t = t.expand(3, -1, -1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(dtype)


@dataclass
class GeneratorConfig:
    output_dir: str = "./generated"
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 0
    negative_prompt: str = "blurry, distorted, noisy, broken geometry"
    save_inputs: bool = False
    max_steps: int = 64


class CADAutoregressiveGenerator:
    """Phase 4 loop: planner → ``CADSingleViewPipeline.generate`` until STOP or cap."""

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

    @staticmethod
    def _load_i_final(i_final: Union[str, Image.Image]) -> Image.Image:
        if isinstance(i_final, str):
            return Image.open(i_final).convert("RGB")
        return i_final.convert("RGB") if i_final.mode != "RGB" else i_final

    def run(
        self,
        i_final: Union[str, Image.Image],
        depth_0_path: Optional[str] = None,
        run_name: str = "run",
    ) -> List[Image.Image]:
        out_dir = os.path.join(self.cfg.output_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)

        i_final_pil = self._load_i_final(i_final)
        i_final_pil.save(os.path.join(out_dir, "i_final_input.png"))

        device = self.pipeline.device
        dtype = self.pipeline.weight_dtype
        i_final_tensor = self._ifinal_tx(i_final_pil).unsqueeze(0).to(device, dtype=dtype)

        if depth_0_path is not None and os.path.isfile(depth_0_path):
            d0 = _load_depth_png_as_rgb_pil(depth_0_path)
            state_pil = d0
            cond = _pil_to_controlnet_tensor(d0, device=device, dtype=dtype)
        else:
            state_pil = Image.new("RGB", (TRAIN_IMAGE_W, TRAIN_IMAGE_H), color=(0, 0, 0))
            cond = _zero_condition(1, device=device, dtype=dtype)

        generator = torch.Generator(device=device).manual_seed(self.cfg.seed)
        produced: List[Image.Image] = []

        for k in range(self.cfg.max_steps):
            prompt = self.planner.predict_next(i_final_pil, state_pil).strip()
            with open(os.path.join(out_dir, "plan_stream.txt"), "a", encoding="utf-8") as fp:
                fp.write(f"[{k}] {prompt}\n")

            if not prompt or PLANNER_STOP_TOKEN in prompt:
                break

            if self.cfg.save_inputs:
                _save_image_tensor(cond[0], os.path.join(out_dir, f"step_{k:03d}_controlnet_cond.png"))

            output = self.pipeline.generate(
                i_final=i_final_tensor,
                condition_image=cond,
                prompt=prompt,
                negative_prompt=self.cfg.negative_prompt,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                generator=generator,
            )
            g_k = output.images[0]
            g_k.save(os.path.join(out_dir, f"step_{k:03d}_overlay.png"))
            produced.append(g_k)

            # // Phase 4 MVP: without CAD boolean + re-render, chain planner state on RGB overlay
            # and approximate the next ControlNet map from its luminance.
            state_pil = g_k
            cond = _pil_to_controlnet_tensor(g_k, device=device, dtype=dtype)

        return produced


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 autoregressive CAD reverse loop (MVP).")
    p.add_argument("--i-final", type=str, required=True, help="Path to final_snapshot.png")
    p.add_argument(
        "--depth-0",
        type=str,
        default=None,
        help="Initial clean depth map (e.g. main_ref_depth.png). If omitted, zeros + black state.",
    )
    p.add_argument("--checkpoint", type=str, default=None, help="train_sd_painter.py output folder (trainables.pt).")
    p.add_argument("--output-dir", type=str, default="./generated")
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--save-inputs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = CADSingleViewPipeline(
        model_cfg=ModelConfig(),
        device=device,
        weight_dtype=infer_dtype,
    ).to_device(device)

    if args.checkpoint is not None:
        pipeline.load_trainables(args.checkpoint, strict=False)

    gen = CADAutoregressiveGenerator(
        pipeline=pipeline,
        planner=MockMLLMPlanner(),
        gen_cfg=GeneratorConfig(
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            save_inputs=args.save_inputs,
            max_steps=args.max_steps,
        ),
    )
    images = gen.run(i_final=args.i_final, depth_0_path=args.depth_0, run_name=args.run_name)
    print(
        f"Phase 4: {len(images)} overlays saved under "
        f"{os.path.join(args.output_dir, args.run_name)}"
    )


if __name__ == "__main__":
    main()