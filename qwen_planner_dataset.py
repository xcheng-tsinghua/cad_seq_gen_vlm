"""Phase 3 — Qwen2.5-VL planner supervision rows (no ``operation_param.json``).

Each sample teaches: given ``final_snapshot.png`` and a **current state** image,
predict the **text prompt for the next modeling step** (the painter instruction
stored in the *next* step's :data:`config.PROMPT_FILENAME`).

``prev_state_mode`` controls how "current state" is defined:

* ``"depth_before_next"`` — ``prev_depth_map.png`` inside the *next* step folder
  (geometry immediately before that step's operation).
* ``"overlay_after_prev"`` — ``overlayed_all.png`` from the *previous* step in the
  sorted sequence (simulates painter canvas after the last op). The first step
  uses :data:`config.MAIN_REF_DEPTH_FILENAME` at part root if present, else that
  step's ``prev_depth_map.png``.

// MVP: single-view ``*_PPP/`` trees only. No JSON in inputs — match Phase 4 where
the planner must infer without CAD parameter dumps.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from torch.utils.data import Dataset

from config import (
    I_FINAL_FILENAME,
    MAIN_REF_DEPTH_FILENAME,
    MVP_VIEW_SUFFIX,
    OVERLAYED_FILENAME,
    PREV_DEPTH_FILENAME,
    PROMPT_FILENAME,
    STEP_DIR_PREFIX,
)

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(rf"^{re.escape(STEP_DIR_PREFIX)}(\d+)$")
_PART_DIR_RE = re.compile(rf"^(?P<part>.+)_{re.escape(MVP_VIEW_SUFFIX)}$")

PrevStateMode = Literal["depth_before_next", "overlay_after_prev"]


@dataclass(frozen=True)
class _PartRecord:
    part_id: str
    sorted_indices: Tuple[int, ...]


class QwenPlannerSFTDataset(Dataset):
    """One row per transition: (I_final, prev_state) → caption for the next step."""

    def __init__(
        self,
        data_root: str,
        prev_state_mode: PrevStateMode = "depth_before_next",
        part_ids_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.prev_state_mode = prev_state_mode
        wl = self._load_whitelist(part_ids_file)
        self.parts = self._scan_parts(wl)
        self.flat: List[Tuple[str, int, str, str, str]] = self._build_index()
        if not self.flat:
            raise RuntimeError(
                f"No planner SFT rows under {data_root!r}. "
                f"Need {I_FINAL_FILENAME}, valid prev-state images, "
                f"and non-empty {PROMPT_FILENAME} from Phase 1 (see prev_state_mode)."
            )
        logger.info(
            "QwenPlannerSFTDataset: %d parts, %d transitions, mode=%s.",
            len(self.parts),
            len(self.flat),
            prev_state_mode,
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
            m = _PART_DIR_RE.match(name)
            if not m:
                continue
            pid = m.group("part")
            if whitelist is not None and pid not in whitelist:
                continue
            idxs = self._discover_step_indices(full)
            if not idxs:
                logger.warning("Part %s: no roll_back steps; skipping.", pid)
                continue
            parts[pid] = _PartRecord(part_id=pid, sorted_indices=idxs)
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

    def _part_root(self, part_id: str) -> str:
        return os.path.join(self.data_root, f"{part_id}_{MVP_VIEW_SUFFIX}")

    def _step_dir(self, part_id: str, idx: int) -> str:
        return os.path.join(self._part_root(part_id), f"{STEP_DIR_PREFIX}{idx}")

    def _read_prompt(self, part_id: str, idx: int) -> Optional[str]:
        path = os.path.join(self._step_dir(part_id, idx), PROMPT_FILENAME)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fp:
            t = fp.read().strip()
        return t or None

    def _resolve_prev_state_path(
        self,
        part_id: str,
        prev_idx: int,
        next_idx: int,
    ) -> Optional[str]:
        if self.prev_state_mode == "depth_before_next":
            p = os.path.join(self._step_dir(part_id, next_idx), PREV_DEPTH_FILENAME)
            return os.path.abspath(p) if os.path.isfile(p) else None
        # overlay_after_prev
        main_ref = os.path.join(self._part_root(part_id), MAIN_REF_DEPTH_FILENAME)
        if prev_idx < 0:
            if os.path.isfile(main_ref):
                return os.path.abspath(main_ref)
            p0 = os.path.join(self._step_dir(part_id, next_idx), PREV_DEPTH_FILENAME)
            return os.path.abspath(p0) if os.path.isfile(p0) else None
        ov = os.path.join(self._step_dir(part_id, prev_idx), OVERLAYED_FILENAME)
        return os.path.abspath(ov) if os.path.isfile(ov) else None

    def _build_index(self) -> List[Tuple[str, int, str, str, str]]:
        """Rows: (part_id, next_step_index, i_final_path, prev_state_path, target_text)."""
        flat: List[Tuple[str, int, str, str, str]] = []
        for part_id, rec in self.parts.items():
            idxs = rec.sorted_indices
            ifinal = os.path.join(self._part_root(part_id), I_FINAL_FILENAME)
            if not os.path.isfile(ifinal):
                logger.warning("Skipping part %s: missing %s", part_id, I_FINAL_FILENAME)
                continue
            ifinal_abs = os.path.abspath(ifinal)
            for t in range(len(idxs)):
                next_idx = idxs[t]
                prev_seq_idx = t - 1
                prev_roll = idxs[prev_seq_idx] if prev_seq_idx >= 0 else -1
                st_path = self._resolve_prev_state_path(part_id, prev_roll, next_idx)
                if st_path is None:
                    logger.warning(
                        "Skipping %s step %d: no prev_state image (mode=%s).",
                        part_id,
                        next_idx,
                        self.prev_state_mode,
                    )
                    continue
                tgt = self._read_prompt(part_id, next_idx)
                if not tgt:
                    logger.warning(
                        "Skipping %s step %d: empty/missing %s — run Phase 1 auto_label.",
                        part_id,
                        next_idx,
                        PROMPT_FILENAME,
                    )
                    continue
                flat.append((part_id, next_idx, ifinal_abs, st_path, tgt))
        return flat

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, i: int) -> Dict[str, object]:
        part_id, next_idx, ifinal, pstate, text = self.flat[i]
        return {
            "part_id": part_id,
            "next_step_index": next_idx,
            "i_final_path": ifinal,
            "prev_state_path": pstate,
            "target_prompt": text,
        }


def collate_planner_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "part_id": [b["part_id"] for b in batch],
        "next_step_index": [b["next_step_index"] for b in batch],
        "i_final_path": [b["i_final_path"] for b in batch],
        "prev_state_path": [b["prev_state_path"] for b in batch],
        "target_prompt": [b["target_prompt"] for b in batch],
    }
