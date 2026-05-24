from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from vision_cad_emu35.data.transforms import load_preprocessed_image
from vision_cad_emu35.utils.jsonl import read_jsonl


SYSTEM_PROMPT = (
    "You are a CAD modeling planner. Given the final CAD part snapshot and the previous "
    "modeling state depth map, infer the current modeling operation type and generate "
    "the CAD-style preview image for this step."
)

USER_PROMPT = """Image 1 is the final CAD part snapshot. Image 2 is the previous state depth map. Based on these two images, predict the current modeling operation.

Preview drawing rules:
- Keep the previous state depth map unchanged as the background.
- Apply a semi-transparent yellow mask on the sketch reference plane.
- Apply a semi-transparent cyan mask on reference geometry such as revolve axis or sweep path.
- Draw the colored incremental wireframe showing the local entity created, modified, or removed in this step.
- Draw red solid lines for the reference 2D sketch used in the current operation.
- Draw blue solid lines for the termination face contour of the local entity.
- Draw green solid lines for edges of the newly added solid entity in this step.
- Draw magenta solid lines for edges of the entity cut or removed in this step.
Return the operation type first, then generate the preview image."""


def assistant_text(operation_type: str) -> str:
    return f"Operation_Type: {operation_type}"


class CADStepDataset:
    """Manifest-backed dataset that keeps model-specific encoding out of the data layer."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 512,
        load_images: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = list(read_jsonl(self.manifest_path))
        self.image_size = image_size
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        prompt = row.get("prompt") or USER_PROMPT
        row["system_prompt"] = row.get("system_prompt") or SYSTEM_PROMPT
        row["prompt"] = prompt
        row["assistant_text"] = assistant_text(row["operation_type"])

        if self.load_images:
            row["final_snapshot"] = load_preprocessed_image(row["final_snapshot_path"], self.image_size)
            row["prev_depth_map"] = load_preprocessed_image(row["prev_depth_map_path"], self.image_size)
            row["target_image"] = load_preprocessed_image(
                row.get("overlayed_all_path", ""),
                self.image_size,
                allow_blank=True,
            )
        return row


def pil_to_rgb_tuple(image: Image.Image) -> tuple[int, int]:
    return image.size

