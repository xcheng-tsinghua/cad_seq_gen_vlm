from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_cad_emu35.inference.executor import Executor, NotImplementedExecutor
from vision_cad_emu35.models.emu35_adapter import Emu35Adapter
from vision_cad_emu35.utils.image_io import load_image_rgb, save_image


class AutoregressiveCADPlanner:
    def __init__(
        self,
        adapter: Emu35Adapter,
        executor: Executor | None = None,
        generation_config: Any | None = None,
    ) -> None:
        self.adapter = adapter
        self.executor = executor or NotImplementedExecutor()
        self.generation_config = generation_config

    def run(
        self,
        final_snapshot_path: str | Path,
        initial_depth_map_path: str | Path,
        output_dir: str | Path,
        max_steps: int = 20,
        prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        final_snapshot_path = Path(final_snapshot_path)
        current_depth_path = Path(initial_depth_map_path)
        steps: list[dict[str, Any]] = []

        for step_index in range(max_steps):
            step_dir = out / f"step_{step_index:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_image(load_image_rgb(current_depth_path), step_dir / "prev_depth_map.png")
            request = {
                "final_snapshot_path": str(final_snapshot_path),
                "prev_depth_map_path": str(current_depth_path),
                "step_index": step_index,
                "prompt": prompt,
            }
            (step_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")

            result = self.adapter.generate(
                load_image_rgb(final_snapshot_path),
                load_image_rgb(current_depth_path),
                prompt,
                self.generation_config,
            )
            operation_type = result["operation_type"]
            save_image(result["image"], step_dir / "overlayed_all.png")
            (step_dir / "operation_type.txt").write_text(operation_type, encoding="utf-8")

            response = {
                "operation_type": operation_type,
                "raw_text": result.get("raw_text", ""),
                "metadata": result.get("metadata", {}),
            }
            (step_dir / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
            steps.append({**response, "step_dir": str(step_dir)})

            if operation_type == "<STOP>":
                break

            next_depth_path = step_dir / "generated_depth_map.png"
            self.executor.run_step(
                final_snapshot_path,
                current_depth_path,
                step_dir / "overlayed_all.png",
                operation_type,
                next_depth_path,
            )
            current_depth_path = next_depth_path

        (out / "sequence.json").write_text(json.dumps(steps, indent=2, default=str), encoding="utf-8")
        return steps

