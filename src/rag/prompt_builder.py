from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from config import RagConfig
from rag.example_formatter import format_reference_example
from utils.image_io import load_image_rgb, resize_pad_image


SYSTEM_PROMPT = (
    "You are a CAD modeling planner. You infer the current CAD modeling step from visual evidence. "
    "You use retrieved historical CAD modeling examples as references."
)


DRAWING_RULES = """Preview drawing rules:
- Keep the previous state depth map with edge frame unchanged as the background.
- Apply a semi-transparent yellow mask on the sketch reference plane.
- Apply a semi-transparent cyan mask on reference geometry such as revolve axis or sweep path.
- Draw the colored_incremental_wireframe showing the local entity created, modified, or removed in this step.
- colored_incremental_wireframe part1: Draw red solid lines for the reference 2D sketch used in the current operation.
- colored_incremental_wireframe part2: Draw blue solid lines for the termination face contour of the local entity.
- colored_incremental_wireframe part3: Draw green solid lines for other edges of the local entity."""


@dataclass
class RagPrompt:
    prompt_text: str
    images: list[Image.Image]
    image_roles: list[str]
    zero_shot: bool
    retrieved_examples: list[dict[str, Any]] = field(default_factory=list)


class RagPromptBuilder:
    def __init__(self, config: RagConfig | None = None, image_size: int = 512) -> None:
        self.config = config or RagConfig()
        self.image_size = image_size

    def build(
        self,
        final_snapshot: Image.Image,
        prev_depth_map: Image.Image,
        retrieved_examples: list[dict[str, Any]],
        prompt_extra: str | None = None,
        operation_type_hint: str | None = None,
    ) -> RagPrompt:
        query_final = resize_pad_image(final_snapshot, self.image_size)
        query_prev = resize_pad_image(prev_depth_map, self.image_size)
        images = [query_final, query_prev]
        image_roles = ["query_final_snapshot", "query_prev_depth_map"]

        refs_text: list[str] = []
        reference_image_count = 0
        included_examples: list[dict[str, Any]] = []
        for idx, example in enumerate(retrieved_examples, start=1):
            refs_text.append(format_reference_example(example, idx))
            packed_roles = []
            for key, role, enabled in (
                ("final_snapshot_path", "reference_final_snapshot", self.config.include_example_final_snapshot),
                ("prev_depth_map_path", "reference_prev_depth_map", self.config.include_example_prev_depth_map),
                ("overlayed_all_path", "reference_overlayed_all", self.config.include_example_overlayed_all and self.config.include_example_outputs),
            ):
                if not enabled or reference_image_count >= self.config.max_reference_images:
                    continue
                path = example.get(key)
                if path and Path(path).exists():
                    images.append(resize_pad_image(load_image_rgb(path), self.image_size))
                    image_roles.append(f"{role}_{idx}")
                    packed_roles.append(role)
                    reference_image_count += 1
            item = dict(example)
            item["packed_image_roles"] = packed_roles
            included_examples.append(item)

        zero_shot = len(retrieved_examples) == 0
        if zero_shot:
            refs = "No retrieved examples are available. Run in zero-shot mode using only the query images and the drawing rules."
        else:
            refs = "\n\n".join(refs_text)

        hint = (
            f"\n\nOperation type hint from user or retriever filter: {operation_type_hint.strip()}"
            if operation_type_hint and operation_type_hint.strip()
            else ""
        )
        extra = f"\n\nAdditional user guidance:\n{prompt_extra.strip()}" if prompt_extra and prompt_extra.strip() else ""
        prompt_text = f"""{SYSTEM_PROMPT}

User:
Image 1 is the final CAD part snapshot for the query.
Image 2 is the previous state depth map with edge frame for the query.

Your task:
Predict the current modeling operation type and generate the CAD-style preview image for this step.

{DRAWING_RULES}

Retrieved reference examples:
{refs}

Please use the retrieved examples only as references. Do not copy them blindly.

Return the operation type first in this exact format:
Operation_Type: <operation_type>

Then generate one preview image.{hint}{extra}"""
        return RagPrompt(
            prompt_text=prompt_text,
            images=images,
            image_roles=image_roles,
            zero_shot=zero_shot,
            retrieved_examples=included_examples,
        )
