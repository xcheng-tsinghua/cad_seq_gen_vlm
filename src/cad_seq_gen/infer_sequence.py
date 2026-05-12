from __future__ import annotations

from pathlib import Path

import torch
import typer
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UNet2DConditionModel
from peft import PeftModel
from PIL import Image

from src.cad_seq_gen.models.step_count import StepCountPredictor
from src.cad_seq_gen.utils.image_ops import (
    make_condition_canvas,
    save_step_images,
    split_step_canvas,
)

app = typer.Typer(add_completion=False)


def _load_pipeline(
    pretrained_model: str,
    controlnet_model: str,
    lora_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> StableDiffusionControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    base_unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    peft_unet = PeftModel.from_pretrained(base_unet, lora_dir / "unet_lora")
    peft_unet = peft_unet.merge_and_unload()
    pipe.unet = peft_unet.to(dtype=dtype)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@app.command()
def main(
    input_image: Path = typer.Option(..., help="User input CAD part image."),
    processed_root: Path = typer.Option(..., help="Prepared dataset root."),
    pretrained_model: str = typer.Option(..., help="Base SD model path or HF id."),
    controlnet_model: str = typer.Option(..., help="ControlNet model path or HF id."),
    lora_dir: Path = typer.Option(..., help="Directory containing unet_lora."),
    output_dir: Path = typer.Option(..., help="Output sequence directory."),
    image_size: int = typer.Option(512),
    num_steps: int = typer.Option(
        0,
        help="If 0 then auto-predict step count; otherwise use this value.",
    ),
    inference_steps_per_frame: int = typer.Option(30),
    guidance_scale: float = typer.Option(5.5),
    seed: int = typer.Option(123),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    output_dir.mkdir(parents=True, exist_ok=True)
    input_img = Image.open(input_image).convert("RGB").resize((image_size, image_size))

    if num_steps <= 0:
        predictor = StepCountPredictor(device=str(device))
        predictor.fit(processed_root=processed_root)
        num_steps = predictor.predict(input_img, min_steps=1, max_steps=40)

    pipe = _load_pipeline(
        pretrained_model=pretrained_model,
        controlnet_model=controlnet_model,
        lora_dir=lora_dir,
        dtype=dtype,
        device=device,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    prev_canvas = None

    for i in range(1, num_steps + 1):
        control_image = make_condition_canvas(
            part_image=input_img,
            prev_canvas=prev_canvas,
            panel_size=image_size,
        )
        out = pipe(
            prompt="cad modeling step canvas, black background, high contrast",
            image=control_image,
            num_inference_steps=inference_steps_per_frame,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        step_canvas = out.images[0]
        step_images = split_step_canvas(step_canvas)
        step_dir = output_dir / f"step_{i:03d}"
        save_step_images(step_images, step_dir)
        prev_canvas = step_canvas

    typer.echo(f"Generated {num_steps} steps to: {output_dir}")


if __name__ == "__main__":
    app()

