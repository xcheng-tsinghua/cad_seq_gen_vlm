"""Dataset loader for the Vision-Based CAD Modeling Step Reverse Generation System."""

from __future__ import annotations

import logging
import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF

from config import (
    I_FINAL_FILENAME,
    PREV_DEPTH_FILENAME,
    STEP_DIR_PREFIX,
)

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(rf"^{re.escape(STEP_DIR_PREFIX)}(\d+)$")


@dataclass(frozen=True)
class _PartRecord:
    part_id: str
    view_suffix: str
    sorted_indices: Tuple[int, ...]


class MLLMPlannerSFTDataset(Dataset):
    """Dataset for MLLM planner SFT. Loads final_snapshot, prev_depth_map, and instruction JSON."""

    def __init__(
        self,
        data_root: str,
        contour_size: Tuple[int, int] = (100, 100),
        part_ids_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.contour_size = contour_size
        wl = self._load_whitelist(part_ids_file)
        self.parts = self._scan_parts(wl)
        self.flat = self._build_index()
        
        if not self.flat:
            raise RuntimeError(
                f"No planner SFT rows under {data_root!r}. "
                f"Need {I_FINAL_FILENAME}, {PREV_DEPTH_FILENAME}, and instruction.json."
            )
        logger.info(
            "MLLMPlannerSFTDataset: %d parts, %d transitions.",
            len(self.parts),
            len(self.flat),
        )

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
            if "_" not in name:
                continue
            pid, view = name.rsplit("_", 1)
            if whitelist is not None and pid not in whitelist:
                continue
            idxs = self._discover_step_indices(full)
            if not idxs:
                continue
            parts[name] = _PartRecord(part_id=pid, view_suffix=view, sorted_indices=idxs)
        return parts

    @staticmethod
    def _discover_step_indices(view_dir: str) -> Tuple[int, ...]:
        out: List[int] = []
        for entry in os.listdir(view_dir):
            mm = _STEP_RE.match(entry)
            if mm and os.path.isdir(os.path.join(view_dir, entry)):
                out.append(int(mm.group(1)))
        out.sort()
        return tuple(out)

    def _build_index(self) -> List[Tuple[str, str, int]]:
        """Rows: (part_dir_name, part_id, step_index)."""
        flat: List[Tuple[str, str, int]] = []
        for name, rec in self.parts.items():
            for idx in rec.sorted_indices:
                flat.append((name, rec.part_id, idx))
        return flat

    def __len__(self) -> int:
        return len(self.flat)

    @staticmethod
    def quantize_coordinates(coords, max_val=100):
        out = []
        for pt in coords:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            x, y = pt[0], pt[1]
            if x > 100 or y > 100:
                x = int(round(x / 10.0))
                y = int(round(y / 10.0))
            x = max(0, min(max_val, x))
            y = max(0, min(max_val, y))
            out.append([x, y])
        return out

    @staticmethod
    def quantize_bbox(bbox, max_val=100):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return [0, 0, 0, 0]
        out = []
        for val in bbox:
            if val > 100:
                val = int(round(val / 10.0))
            val = max(0, min(max_val, val))
            out.append(val)
        return out

    @staticmethod
    def serialize_instruction_with_spans(inst: dict) -> Tuple[str, dict]:
        spans = {
            "sketch (red)": [],
            "other_counter (green / magenta)": [],
            "terminate_face_contour (blue)": [],
            "sketch_plane_contour (yellow)": [],
            "reference_geom_contour (cyan)": [],
            "target_region_bbox (red)": []
        }
        
        curr_str = "{\n"
        curr_str += f'    "operation_type": "{inst["operation_type"]}",\n'
        
        for key in ["sketch (red)", "other_counter (green / magenta)", "terminate_face_contour (blue)", "sketch_plane_contour (yellow)", "reference_geom_contour (cyan)"]:
            curr_str += f'    "{key}": ['
            coords = inst.get(key, [])
            for idx, (x, y) in enumerate(coords):
                curr_str += "["
                
                # Record span for x
                x_str = str(x)
                start_x = len(curr_str)
                curr_str += x_str
                end_x = len(curr_str)
                spans[key].append((start_x, end_x))
                
                curr_str += ","
                
                # Record span for y
                y_str = str(y)
                start_y = len(curr_str)
                curr_str += y_str
                end_y = len(curr_str)
                spans[key].append((start_y, end_y))
                
                curr_str += "]"
                if idx < len(coords) - 1:
                    curr_str += ", "
            curr_str += "],\n"
            
        curr_str += '    "target_region_bbox (red)": ['
        bbox = inst.get("target_region_bbox (red)", [])
        for idx, val in enumerate(bbox):
            val_str = str(val)
            start_val = len(curr_str)
            curr_str += val_str
            end_val = len(curr_str)
            spans["target_region_bbox (red)"].append((start_val, end_val))
            if idx < len(bbox) - 1:
                curr_str += ", "
        curr_str += "]\n}"
        
        return curr_str, spans

    def __getitem__(self, i: int) -> Dict[str, object]:
        part_name, part_id, idx = self.flat[i]
        part_dir = os.path.join(self.data_root, part_name)
        step_dir = os.path.join(part_dir, f"{STEP_DIR_PREFIX}{idx}")
        
        final_snapshot_path = os.path.join(part_dir, I_FINAL_FILENAME)
        prev_depth_path = os.path.join(step_dir, PREV_DEPTH_FILENAME)
        instruction_path = os.path.join(step_dir, "instruction.json")
        
        # Load and parse instruction.json
        if os.path.isfile(instruction_path):
            try:
                with open(instruction_path, "r", encoding="utf-8") as f:
                    raw_inst = json.load(f)
            except Exception as e:
                logger.error(f"Error loading {instruction_path}: {e}")
                raw_inst = {}
        else:
            raw_inst = {}
            
        inst = {}
        inst["operation_type"] = raw_inst.get("operation_type", "none")
        inst["sketch (red)"] = self.quantize_coordinates(raw_inst.get("sketch (red)", raw_inst.get("sketch", [])))
        inst["other_counter (green / magenta)"] = self.quantize_coordinates(raw_inst.get("other_counter (green / magenta)", raw_inst.get("other_counter", [])))
        inst["terminate_face_contour (blue)"] = self.quantize_coordinates(raw_inst.get("terminate_face_contour (blue)", raw_inst.get("terminate_face_contour", [])))
        inst["sketch_plane_contour (yellow)"] = self.quantize_coordinates(raw_inst.get("sketch_plane_contour (yellow)", raw_inst.get("sketch_plane_contour", [])))
        inst["reference_geom_contour (cyan)"] = self.quantize_coordinates(raw_inst.get("reference_geom_contour (cyan)", raw_inst.get("reference_geom_contour", [])))
        inst["target_region_bbox (red)"] = self.quantize_bbox(raw_inst.get("target_region_bbox (red)", raw_inst.get("target_region_bbox", [0, 0, 0, 0])))
        
        # Serialize to formatted instruction JSON string and get spans
        inst_text, spans = self.serialize_instruction_with_spans(inst)
        
        # Load the 5 contour PNGs
        contours = {}
        contour_filenames = {
            "sketch": "sketch.png",
            "other_counter": "other_counter.png",
            "terminate_face_contour": "terminate_face_contour.png",
            "sketch_plane_contour": "sketch_plane_contour.png",
            "reference_geom_contour": "reference_geom_contour.png"
        }
        
        for key, fname in contour_filenames.items():
            cpath = os.path.join(step_dir, fname)
            if os.path.isfile(cpath):
                try:
                    with Image.open(cpath) as img:
                        img_gray = img.convert("L").resize(self.contour_size, Image.Resampling.BILINEAR)
                        t = TF.to_tensor(img_gray)
                except Exception as e:
                    logger.error(f"Error reading {cpath}: {e}")
                    t = torch.zeros(1, self.contour_size[0], self.contour_size[1], dtype=torch.float32)
            else:
                t = torch.zeros(1, self.contour_size[0], self.contour_size[1], dtype=torch.float32)
            contours[key] = t
            
        return {
            "part_id": part_id,
            "step_index": idx,
            "final_snapshot_path": final_snapshot_path,
            "prev_depth_path": prev_depth_path,
            "instruction": inst,
            "instruction_text": inst_text,
            "spans": spans,
            "contours": contours,
        }


def collate_planner_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    res = {
        "part_id": [b["part_id"] for b in batch],
        "step_index": [b["step_index"] for b in batch],
        "final_snapshot_path": [b["final_snapshot_path"] for b in batch],
        "prev_depth_path": [b["prev_depth_path"] for b in batch],
        "instruction": [b["instruction"] for b in batch],
        "instruction_text": [b["instruction_text"] for b in batch],
        "spans": [b["spans"] for b in batch],
    }
    
    # Collate contours
    contour_keys = ["sketch", "other_counter", "terminate_face_contour", "sketch_plane_contour", "reference_geom_contour"]
    res["contours"] = {
        k: torch.stack([b["contours"][k] for b in batch], dim=0)
        for k in contour_keys
    }
    return res
