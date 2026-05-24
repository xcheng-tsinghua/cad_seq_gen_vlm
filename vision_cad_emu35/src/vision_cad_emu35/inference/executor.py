from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class Executor(ABC):
    """Abstract interface for downstream CAD execution/rendering."""

    @abstractmethod
    def run_step(
        self,
        final_snapshot_path: str | Path,
        prev_depth_map_path: str | Path,
        overlayed_all_path: str | Path,
        operation_type: str,
        output_depth_map_path: str | Path,
    ) -> Path:
        raise NotImplementedError


class NotImplementedExecutor(Executor):
    def run_step(
        self,
        final_snapshot_path: str | Path,
        prev_depth_map_path: str | Path,
        overlayed_all_path: str | Path,
        operation_type: str,
        output_depth_map_path: str | Path,
    ) -> Path:
        raise NotImplementedError(
            "No CAD executor is configured. For novel parts, connect a CAD parser/renderer "
            "that consumes overlayed_all.png and operation_type and writes depth_k."
        )


class CopyDepthExecutor(Executor):
    """Teacher-forced executor that copies a known current_depth_map sequence."""

    def __init__(self, depth_sequence: list[str | Path]) -> None:
        self.depth_sequence = [Path(p) for p in depth_sequence]
        self.index = 0

    def run_step(
        self,
        final_snapshot_path: str | Path,
        prev_depth_map_path: str | Path,
        overlayed_all_path: str | Path,
        operation_type: str,
        output_depth_map_path: str | Path,
    ) -> Path:
        if self.index >= len(self.depth_sequence):
            raise IndexError("Teacher-forced depth sequence is exhausted.")
        source = self.depth_sequence[self.index]
        self.index += 1
        target = Path(output_depth_map_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

