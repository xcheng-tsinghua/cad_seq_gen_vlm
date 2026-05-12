from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class ControlNetStepDataset(Dataset):
    def __init__(self, processed_root: Path, split: str = "train") -> None:
        self.processed_root = Path(processed_root)
        self.split = split
        self.rows: List[Dict] = []
        self._load()

    def _load(self) -> None:
        manifest = self.processed_root / "manifest.jsonl"
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["split"] == self.split:
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        target = Image.open(row["target_image"]).convert("RGB")
        control = Image.open(row["control_image"]).convert("RGB")
        return {
            "pixel_values": _to_tensor(target),
            "control_values": _to_tensor(control),
            "prompt": row["prompt"],
            "sample_id": row["sample_id"],
        }

