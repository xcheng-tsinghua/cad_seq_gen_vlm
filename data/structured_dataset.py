from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.image_ops import parse_roll_back_index

KEYS = ("prev_depth_map", "sketch_plane_mask", "reference_mask", "result_frame")


def _resize_keep_aspect_with_pad(path: str, image_size: int) -> np.ndarray:
    """Load grayscale image with aspect-ratio-preserving resize + zero padding."""
    img = Image.open(path).convert("L")
    src_w, src_h = img.size
    scale = min(image_size / max(src_w, 1), image_size / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    x0 = (image_size - new_w) // 2
    y0 = (image_size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = np.array(resized).astype(np.float32) / 255.0
    return canvas


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


class StructuredStepDataset(Dataset):
    """Input channels:
    [target_part, prev_depth, prev_sketch_mask, prev_reference_mask, prev_result_frame, target_part_edges]
    """

    def __init__(
        self,
        processed_root: Path | None = None,
        raw_root: Path | None = None,
        split: str = "train",
        image_size: int = 512,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        if processed_root is None and raw_root is None:
            raise ValueError("Either processed_root or raw_root must be provided.")
        self.processed_root = Path(processed_root) if processed_root is not None else None
        self.raw_root = Path(raw_root) if raw_root is not None else None
        self.split = split
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.rows: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if self.processed_root is not None and (self.processed_root / "manifest.jsonl").exists():
            manifest = self.processed_root / "manifest.jsonl"
            with manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row["split"] == self.split:
                        self.rows.append(row)
            return

        if self.raw_root is None:
            raise FileNotFoundError("manifest.jsonl not found and raw_root is not provided.")

        self.rows = self._build_rows_from_raw()

    def _collect_steps(self, part_dir: Path) -> List[Path]:
        step_dirs = [
            p for p in part_dir.iterdir() if p.is_dir() and p.name.startswith("roll_back_index_")
        ]
        step_dirs.sort(key=lambda p: parse_roll_back_index(p.name))
        return step_dirs

    def _step_paths(self, step_dir: Path) -> Dict[str, str]:
        return {
            "prev_depth_map": str((step_dir / "prev_depth_map.png").as_posix()),
            "sketch_plane_mask": str((step_dir / "sketch_plane_mask.png").as_posix()),
            "reference_mask": str((step_dir / "reference_mask.png").as_posix()),
            "result_frame": str((step_dir / "result_frame.png").as_posix()),
        }

    def _build_rows_from_raw(self) -> List[Dict]:
        rng = random.Random(self.seed)
        part_dirs = [p for p in self.raw_root.iterdir() if p.is_dir()]
        part_dirs.sort(key=lambda p: p.name)
        rng.shuffle(part_dirs)
        val_count = max(1, int(len(part_dirs) * self.val_ratio)) if part_dirs else 0
        val_part_ids = {p.name for p in part_dirs[:val_count]}

        rows: List[Dict] = []
        for part_dir in sorted(part_dirs, key=lambda p: p.name):
            step_dirs = self._collect_steps(part_dir)
            if not step_dirs:
                continue
            part_id = part_dir.name
            split = "val" if part_id in val_part_ids else "train"
            if split != self.split:
                continue
            final_result_frame = str((step_dirs[-1] / "result_frame.png").as_posix())

            for seq_idx, step_dir in enumerate(step_dirs, start=1):
                rows.append(
                    {
                        "sample_id": f"{part_id}__step_{seq_idx:03d}",
                        "part_id": part_id,
                        "step_index": seq_idx,
                        "split": split,
                        "target_part_image": final_result_frame,
                        "target": self._step_paths(step_dir),
                        "prev_target": self._step_paths(step_dirs[seq_idx - 2]) if seq_idx > 1 else None,
                    }
                )
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        target_part = _resize_keep_aspect_with_pad(row["target_part_image"], self.image_size)
        edges = cv2.Canny((target_part * 255).astype(np.uint8), 80, 180).astype(np.float32) / 255.0

        if row["prev_target"] is None:
            prev = {k: np.zeros_like(target_part, dtype=np.float32) for k in KEYS}
        else:
            prev = {k: _resize_keep_aspect_with_pad(row["prev_target"][k], self.image_size) for k in KEYS}

        target = {k: _resize_keep_aspect_with_pad(row["target"][k], self.image_size) for k in KEYS}

        x = np.stack(
            [
                target_part,
                prev["prev_depth_map"],
                prev["sketch_plane_mask"],
                prev["reference_mask"],
                prev["result_frame"],
                edges,
            ],
            axis=0,
        )
        y = np.stack(
            [
                target["prev_depth_map"],
                target["sketch_plane_mask"],
                target["reference_mask"],
                target["result_frame"],
            ],
            axis=0,
        )

        return {
            "inputs": _to_tensor(x),
            "targets": _to_tensor(y),
            "sample_id": row["sample_id"],
            "part_id": row["part_id"],
        }

