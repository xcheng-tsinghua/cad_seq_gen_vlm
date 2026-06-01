from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config import AppConfig
from filenames import OUTPUT_GENERATED_DEPTH_MAP
from inference.rag_single_step import run_rag_single_step
from models.emu35_adapter import Emu35Adapter
from rag.retriever import RagRetriever


class AutoregressiveRagPlanner:
    """Skeleton loop for frozen Emu3.5 RAG planning.

    A real CAD executor should replace the default no-op depth propagation.
    """

    def __init__(self, config: AppConfig, adapter: Emu35Adapter, retriever: RagRetriever) -> None:
        self.config = config
        self.adapter = adapter
        self.retriever = retriever

    def run(
        self,
        final_snapshot_path: str | Path,
        initial_depth_map_path: str | Path,
        output_dir: str | Path,
        max_steps: int = 20,
        prompt_extra: str | None = None,
    ) -> list[dict[str, Any]]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        current_depth = Path(initial_depth_map_path)
        steps = []
        for idx in range(max_steps):
            step_dir = out / f"step_{idx:03d}"
            result = run_rag_single_step(
                self.config,
                self.adapter,
                self.retriever,
                final_snapshot_path,
                current_depth,
                step_dir,
                prompt_extra=prompt_extra,
            )
            steps.append({k: v for k, v in result.items() if k != "image"})
            if result.get("operation_type") == "<STOP>":
                break
            next_depth = step_dir / OUTPUT_GENERATED_DEPTH_MAP
            shutil.copy2(current_depth, next_depth)
            current_depth = next_depth
        (out / "sequence.json").write_text(json.dumps(steps, indent=2, default=str), encoding="utf-8")
        return steps
