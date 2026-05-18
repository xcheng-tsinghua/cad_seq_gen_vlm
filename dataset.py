"""Single-view CAD step dataset — **Phase 2** diffusion training.

// MVP Refactor: linear sequence under ``[PART_ID]_PPP/`` only — no 8-view scan,
no einops grids, no ``G_prev`` multi-tile tensor.

Expected layout::

    data_root/
    └── PART123_PPP/
        ├── final_snapshot.png
        ├── roll_back_index_1/
        │   ├── prev_depth_map.png
        │   ├── overlayed_all.png
        │   └── prompt.txt
        └── ...
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import (
    IFINAL_H,
    IFINAL_W,
    I_FINAL_FILENAME,
    MVP_VIEW_SUFFIX,
    OVERLAYED_FILENAME,
    PREV_DEPTH_FILENAME,
    PROMPT_FILENAME,
    STEP_DIR_PREFIX,
    TRAIN_IMAGE_H,
    TRAIN_IMAGE_W,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PartRecord:
    part_id: str
    sorted_indices: Tuple[int, ...]


@dataclass(frozen=True)
class _StepIndex:
    part_id: str
    sorted_pos: int
    cur_index: int
    prev_index: Optional[int]


_STEP_RE = re.compile(rf"^{re.escape(STEP_DIR_PREFIX)}(\d+)$")
# Directory name must end with _PPP (single MVP view).
_PART_DIR_RE = re.compile(rf"^(?P<part>.+)_{re.escape(MVP_VIEW_SUFFIX)}$")


class CADSingleViewDataset(Dataset):
    """One sample per modeling step: ``I_final``, ``condition``, ``target``, ``prompt``.

    // MVP Refactor: replaces ``CADMultiViewDataset``.
    """

    def __init__(
        self,
        data_root: str,
        train_image_size: Tuple[int, int] = (TRAIN_IMAGE_H, TRAIN_IMAGE_W),
        i_final_size: Tuple[int, int] = (IFINAL_H, IFINAL_W),
        part_ids_file: Optional[str] = None,
        fallback_prompt_template: str = "CAD modeling step {step} for part {part}",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.train_h, self.train_w = train_image_size
        self.i_final_h, self.i_final_w = i_final_size
        self.fallback_prompt_template = fallback_prompt_template
        _ = seed  # reserved — no random view picking in MVP

        whitelist = self._load_whitelist(part_ids_file)
        self.parts: Dict[str, _PartRecord] = self._scan_parts(whitelist)
        self.index: List[_StepIndex] = self._build_flat_index(self.parts)

        if len(self.index) == 0:
            raise RuntimeError(
                f"No (part, step) pairs under '{data_root}'. "
                f"Expect directories like PART_{MVP_VIEW_SUFFIX}/."
            )
        logger.info(
            "CADSingleViewDataset: %d parts, %d steps (MVP view %s).",
            len(self.parts),
            len(self.index),
            MVP_VIEW_SUFFIX,
        )

        # [-1, 1] for VAE / ControlNet
        self._rgb_tx = transforms.Compose([
            transforms.Resize((self.train_h, self.train_w), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self._ifinal_tx = transforms.Compose([
            transforms.Resize((self.i_final_h, self.i_final_w), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def _load_whitelist(path: Optional[str]) -> Optional[set]:
        if path is None:
            return None
        with open(path, "r", encoding="utf-8") as fp:
            lines = (ln.strip() for ln in fp.readlines())
            return {ln for ln in lines if ln and not ln.startswith("#")}

    def _scan_parts(self, whitelist: Optional[set]) -> Dict[str, _PartRecord]:
        parts: Dict[str, _PartRecord] = {}
        for name in sorted(os.listdir(self.data_root)):
            full = os.path.join(self.data_root, name)
            if not os.path.isdir(full):
                continue
            m = _PART_DIR_RE.match(name)
            if not m:
                continue
            part_id = m.group("part")
            if whitelist is not None and part_id not in whitelist:
                continue
            sorted_indices = self._discover_step_indices(full)
            if not sorted_indices:
                logger.warning("Part '%s' has no %s* steps; skipping.", part_id, STEP_DIR_PREFIX)
                continue
            parts[part_id] = _PartRecord(part_id=part_id, sorted_indices=sorted_indices)
        return parts

    @staticmethod
    def _discover_step_indices(view_dir: str) -> Tuple[int, ...]:
        if not os.path.isdir(view_dir):
            return tuple()
        indices: List[int] = []
        for entry in os.listdir(view_dir):
            mm = _STEP_RE.match(entry)
            if mm and os.path.isdir(os.path.join(view_dir, entry)):
                indices.append(int(mm.group(1)))
        indices.sort()
        return tuple(indices)

    @staticmethod
    def _build_flat_index(parts: Dict[str, _PartRecord]) -> List[_StepIndex]:
        flat: List[_StepIndex] = []
        for part_id, record in parts.items():
            for pos, idx in enumerate(record.sorted_indices):
                prev = record.sorted_indices[pos - 1] if pos > 0 else None
                flat.append(
                    _StepIndex(
                        part_id=part_id,
                        sorted_pos=pos,
                        cur_index=idx,
                        prev_index=prev,
                    )
                )
        return flat

    def _part_root(self, part_id: str) -> str:
        return os.path.join(self.data_root, f"{part_id}_{MVP_VIEW_SUFFIX}")

    def _step_dir(self, part_id: str, roll_back_index: int) -> str:
        return os.path.join(
            self._part_root(part_id),
            f"{STEP_DIR_PREFIX}{roll_back_index}",
        )

    def _load_prev_depth_rgb(self, part_id: str, roll_back_index: int) -> torch.Tensor:
        """``prev_depth_map.png`` → ``(3, H, W)`` in ``[-1,1]``, identical RGB channels."""
        path = os.path.join(self._step_dir(part_id, roll_back_index), PREV_DEPTH_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing prev depth: {path}")
        img = Image.open(path).convert("L")
        img = img.resize((self.train_w, self.train_h), Image.Resampling.BILINEAR)
        t = TF.to_tensor(img)
        t = t.expand(3, -1, -1)
        return t * 2.0 - 1.0

    def _zero_condition(self) -> torch.Tensor:
        return torch.zeros(3, self.train_h, self.train_w, dtype=torch.float32)

    def _load_target(self, part_id: str, roll_back_index: int) -> torch.Tensor:
        path = os.path.join(self._step_dir(part_id, roll_back_index), OVERLAYED_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing target overlay: {path}")
        return self._rgb_tx(Image.open(path).convert("RGB"))

    def _load_i_final(self, part_id: str) -> torch.Tensor:
        path = os.path.join(self._part_root(part_id), I_FINAL_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing I_final: {path}")
        return self._ifinal_tx(Image.open(path).convert("RGB"))

    def _load_prompt(self, part_id: str, roll_back_index: int) -> str:
        path = os.path.join(self._step_dir(part_id, roll_back_index), PROMPT_FILENAME)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fp:
                text = fp.read().strip()
            if text:
                return text
        return self.fallback_prompt_template.format(part=part_id, step=roll_back_index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.index[idx]
        i_final = self._load_i_final(rec.part_id)
        target_image = self._load_target(rec.part_id, rec.cur_index)
        if rec.prev_index is None:
            condition_image = self._zero_condition()
        else:
            condition_image = self._load_prev_depth_rgb(rec.part_id, rec.cur_index)
        prompt = self._load_prompt(rec.part_id, rec.cur_index)

        return {
            "I_final": i_final,
            "condition_image": condition_image,
            "target_image": target_image,
            "prompt": prompt,
            "part_id": rec.part_id,
            "sorted_pos": rec.sorted_pos,
            "cur_index": rec.cur_index,
            "prev_index": rec.prev_index if rec.prev_index is not None else -1,
        }


def collate_cad_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "I_final": torch.stack([b["I_final"] for b in batch], dim=0),
        "condition_image": torch.stack([b["condition_image"] for b in batch], dim=0),
        "target_image": torch.stack([b["target_image"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "part_id": [b["part_id"] for b in batch],
        "sorted_pos": [b["sorted_pos"] for b in batch],
        "cur_index": [b["cur_index"] for b in batch],
        "prev_index": [b["prev_index"] for b in batch],
    }


def make_worker_init_fn(base_seed: int):
    """Reserved for future RNG per worker; MVP dataset is deterministic per index."""

    def _init(worker_id: int) -> None:
        return

    return _init
