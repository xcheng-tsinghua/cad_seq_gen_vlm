from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_cad_emu35.config import GenerationConfig
from vision_cad_emu35.train.losses import scalar_loss_value
from vision_cad_emu35.utils.image_io import save_image


def validate_loss(adapter: Any, dataloader: Any, max_batches: int | None = None) -> float:
    import torch

    if hasattr(adapter.model, "eval"):
        adapter.model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_batches is not None and step >= max_batches:
                break
            loss = adapter.forward_loss(batch)
            losses.append(scalar_loss_value(loss))
    if hasattr(adapter.model, "train"):
        adapter.model.train()
    return sum(losses) / len(losses) if losses else 0.0


def save_validation_samples(
    adapter: Any,
    dataset: Any,
    output_dir: str | Path,
    generation_config: GenerationConfig,
    max_samples: int = 4,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for idx in range(min(max_samples, len(dataset))):
        sample = dataset[idx]
        try:
            result = adapter.generate(sample["final_snapshot"], sample["prev_depth_map"], sample["prompt"], generation_config)
        except NotImplementedError:
            return
        sample_dir = out / f"sample_{idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "prediction_operation_type.txt").write_text(result["operation_type"], encoding="utf-8")
        (sample_dir / "target_operation_type.txt").write_text(sample["operation_type"], encoding="utf-8")
        save_image(result["image"], sample_dir / "prediction_overlayed_all.png")
        save_image(sample["target_image"], sample_dir / "target_overlayed_all.png")

