"""CADMultiViewDataset – per-step samples for the autoregressive trainer.

Expected on-disk layout
-----------------------

```
data_root/
├── <CAD_PART_ID>_PPP/                  # view = V3d_XposYposZpos
│   ├── roll_back_index_1/              # modeling step #1 (indices may have gaps)
│   │   ├── prev_depth_map.png
│   │   ├── sketch_plane_mask.png
│   │   ├── reference_mask.png
│   │   ├── result_frame.png
│   │   └── prompt.txt                  # optional: natural-language command
│   ├── roll_back_index_3/              # next available step (note: jumps allowed)
│   │   └── ...
│   ├── final_shape_frame.png
│   └── final_snapshot.png              # candidate `I_final` for this view
├── <CAD_PART_ID>_PPN/                  # view = V3d_XposYposZneg
│   └── ... (same structure)
├── ... (PNP, PNN, NPP, NPN, NNP, NNN)
├── <OTHER_PART_ID>_PPP/
└── ...
```

Key contracts
~~~~~~~~~~~~~

* For every CAD part, **all 8 view suffixes must be present** (configurable
  via ``DataConfig.require_all_views``). The set of available roll-back
  indices is read from the anchor view (``ANCHOR_VIEW_SUFFIX``); we *assume*
  the same set of steps exists in every view folder. If a per-view file is
  missing at ``__getitem__`` time, we raise ``FileNotFoundError`` with a
  precise path.
* ``roll_back_index_N`` values are sorted numerically. "Previous step" =
  the immediately preceding entry in that sorted list (not necessarily
  ``N - 1``). The first sorted index yields an all-zero ``G_prev``.
* ``I_final`` is sampled randomly among the 8 views' ``final_snapshot.png``
  whenever ``DataConfig.random_i_final_view`` is True. Otherwise the anchor
  view is used (deterministic, e.g. for validation).
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    ANCHOR_VIEW_SUFFIX,
    IFINAL_H,
    IFINAL_W,
    I_FINAL_FILENAME,
    NUM_ROWS,
    NUM_VIEWS,
    PROMPT_FILENAME,
    ROW_FILENAMES,
    ROW_NAMES,
    STEP_DIR_PREFIX,
    TILE_H,
    TILE_W,
    VIEW_SUFFIXES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal records. ``_PartRecord`` is built once per CAD part during scan;
# ``_StepIndex`` is one flat dataset item.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _PartRecord:
    part_id: str
    sorted_indices: Tuple[int, ...]   # roll-back indices in sorted order


@dataclass(frozen=True)
class _StepIndex:
    part_id: str
    sorted_pos: int                   # position inside ``_PartRecord.sorted_indices``
    cur_index: int                    # actual roll-back index for this step
    prev_index: Optional[int]         # actual roll-back index of previous step (or None)


_STEP_RE = re.compile(rf"^{re.escape(STEP_DIR_PREFIX)}(\d+)$")
_PART_SUFFIX_RE = re.compile(
    r"^(?P<part>.+)_(?P<suffix>" + "|".join(re.escape(s) for s in VIEW_SUFFIXES) + r")$"
)


# ---------------------------------------------------------------------------
class CADMultiViewDataset(Dataset):
    """Dataset returning ``(I_final, G_prev, G_target, prompt)`` per modeling step.

    Parameters
    ----------
    data_root:
        Root folder containing the ``<CAD_PART_ID>_<SUFFIX>`` directories.
    tile_size:
        ``(H, W)`` of a single sub-image in the 4 x 8 grid.
    i_final_size:
        ``(H, W)`` of the conditioning image.
    random_i_final_view:
        If True, sample a random view's ``final_snapshot.png`` per item.
        If False, always use ``ANCHOR_VIEW_SUFFIX`` (deterministic).
    require_all_views:
        If True, raise on parts that lack any of the 8 view folders;
        otherwise emit a warning and skip them.
    part_ids_file:
        Optional path to a text file listing one ``CAD_PART_ID`` per line.
        When provided, only those parts are loaded (useful for train/val
        splits). ``None`` => scan everything found under ``data_root``.
    fallback_prompt_template:
        Used when a per-step ``prompt.txt`` is missing. ``{part}`` and
        ``{step}`` are substituted with the part id and the actual
        roll-back index respectively.
    seed:
        Optional seed for the per-item random view sampler. ``None`` =>
        sampler is seeded from system entropy (different across DataLoader
        workers).
    """

    def __init__(
        self,
        data_root: str,
        tile_size: Tuple[int, int] = (TILE_H, TILE_W),
        i_final_size: Tuple[int, int] = (IFINAL_H, IFINAL_W),
        random_i_final_view: bool = True,
        require_all_views: bool = True,
        part_ids_file: Optional[str] = None,
        fallback_prompt_template: str = "CAD modeling step {step} for part {part}",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.tile_h, self.tile_w = tile_size
        self.i_final_h, self.i_final_w = i_final_size
        self.random_i_final_view = random_i_final_view
        self.require_all_views = require_all_views
        self.fallback_prompt_template = fallback_prompt_template

        # Independent RNG so we don't perturb other torch / numpy state.
        self._rng = random.Random(seed)

        # ------------------------------------------------------------ scan
        whitelist = self._load_whitelist(part_ids_file)
        self.parts: Dict[str, _PartRecord] = self._scan_parts(whitelist)
        self.index: List[_StepIndex] = self._build_flat_index(self.parts)

        if len(self.index) == 0:
            raise RuntimeError(
                f"No (part, step) pairs found under '{data_root}'. "
                "Check the directory layout against the dataset docstring."
            )
        logger.info(
            "CADMultiViewDataset: %d parts, %d total (part, step) items.",
            len(self.parts), len(self.index),
        )

        # ------------------------------------------------------------ transforms
        # Inputs must land in [-1, 1] for both VAE encoding and ControlNet.
        self._tile_tx = transforms.Compose([
            transforms.Resize((self.tile_h, self.tile_w), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self._ifinal_tx = transforms.Compose([
            transforms.Resize((self.i_final_h, self.i_final_w), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    # =========================================================== scan helpers
    @staticmethod
    def _load_whitelist(path: Optional[str]) -> Optional[set]:
        if path is None:
            return None
        with open(path, "r", encoding="utf-8") as fp:
            lines = (ln.strip() for ln in fp.readlines())
            ids = {ln for ln in lines if ln and not ln.startswith("#")}
        return ids

    def _scan_parts(self, whitelist: Optional[set]) -> Dict[str, _PartRecord]:
        """Walk ``data_root`` once and group folders by CAD part id."""
        # part_id -> set of view suffixes encountered
        encountered: Dict[str, set] = {}
        for name in sorted(os.listdir(self.data_root)):
            full = os.path.join(self.data_root, name)
            if not os.path.isdir(full):
                continue
            m = _PART_SUFFIX_RE.match(name)
            if not m:
                continue
            part_id = m.group("part")
            suffix = m.group("suffix")
            if whitelist is not None and part_id not in whitelist:
                continue
            encountered.setdefault(part_id, set()).add(suffix)

        # Validate per-part view completeness and collect step indices.
        parts: Dict[str, _PartRecord] = {}
        all_suffixes = set(VIEW_SUFFIXES)
        for part_id, suffixes_present in encountered.items():
            missing = all_suffixes - suffixes_present
            if missing:
                msg = (
                    f"Part '{part_id}' is missing view suffix(es): "
                    f"{sorted(missing)}."
                )
                if self.require_all_views:
                    raise RuntimeError(msg + " Set require_all_views=False to skip.")
                logger.warning("%s Skipping part.", msg)
                continue

            anchor_dir = os.path.join(self.data_root, f"{part_id}_{ANCHOR_VIEW_SUFFIX}")
            sorted_indices = self._discover_step_indices(anchor_dir)
            if not sorted_indices:
                logger.warning("Part '%s' has no roll_back_index_* steps. Skipping.", part_id)
                continue
            parts[part_id] = _PartRecord(part_id=part_id, sorted_indices=sorted_indices)

        return parts

    @staticmethod
    def _discover_step_indices(view_dir: str) -> Tuple[int, ...]:
        """Return the sorted tuple of roll-back indices inside one view folder."""
        if not os.path.isdir(view_dir):
            return tuple()
        indices: List[int] = []
        for entry in os.listdir(view_dir):
            m = _STEP_RE.match(entry)
            if m and os.path.isdir(os.path.join(view_dir, entry)):
                indices.append(int(m.group(1)))
        indices.sort()
        return tuple(indices)

    @staticmethod
    def _build_flat_index(parts: Dict[str, _PartRecord]) -> List[_StepIndex]:
        """Flatten the (part, step) grid into a single addressable list.

        We treat "step ordering" as the *position* in the sorted-index list,
        so that ``prev`` always means the previous *available* step,
        regardless of any gaps in the raw roll-back numbering.
        """
        flat: List[_StepIndex] = []
        for part_id, record in parts.items():
            for pos, idx in enumerate(record.sorted_indices):
                prev = record.sorted_indices[pos - 1] if pos > 0 else None
                flat.append(_StepIndex(
                    part_id=part_id, sorted_pos=pos,
                    cur_index=idx, prev_index=prev,
                ))
        return flat

    # =========================================================== loading helpers
    def _step_dir(self, part_id: str, suffix: str, roll_back_index: int) -> str:
        return os.path.join(
            self.data_root,
            f"{part_id}_{suffix}",
            f"{STEP_DIR_PREFIX}{roll_back_index}",
        )

    def _load_step_grid(self, part_id: str, roll_back_index: int) -> torch.Tensor:
        """Load all 32 sub-images of one step (8 views x 4 rows) and tile them.

        Returns
        -------
        torch.Tensor
            Shape ``(3, NUM_ROWS * tile_h, NUM_VIEWS * tile_w)`` in ``[-1, 1]``.
        """
        # Build a (NUM_ROWS, NUM_VIEWS, 3, h, w) tensor first, then tile.
        # We iterate row-major to match einops' "r v c h w -> c (r h) (v w)".
        per_row: List[List[torch.Tensor]] = []
        for row_idx, row_filename in enumerate(ROW_FILENAMES):
            per_view: List[torch.Tensor] = []
            for view_suffix in VIEW_SUFFIXES:
                path = os.path.join(
                    self._step_dir(part_id, view_suffix, roll_back_index),
                    row_filename,
                )
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Missing tile for part={part_id} view={view_suffix} "
                        f"step={roll_back_index} row={ROW_NAMES[row_idx]}: {path}"
                    )
                img = Image.open(path).convert("RGB")
                per_view.append(self._tile_tx(img))                  # (3, h, w)
            per_row.append(torch.stack(per_view, dim=0))             # (NUM_VIEWS, 3, h, w)

        # (NUM_ROWS, NUM_VIEWS, 3, h, w)
        grid_5d = torch.stack(per_row, dim=0)
        # r v c h w -> c (r h) (v w)
        grid = rearrange(grid_5d, "r v c h w -> c (r h) (v w)")
        return grid

    def _zero_grid(self) -> torch.Tensor:
        """All-zero placeholder grid for the first sorted step."""
        # We use exact zeros (mid-gray after de-normalization). The ControlNet's
        # conv_out is zero-initialized, so the very first step receives no
        # geometric signal from G_prev anyway.
        return torch.zeros(
            (3, NUM_ROWS * self.tile_h, NUM_VIEWS * self.tile_w),
            dtype=torch.float32,
        )

    def _pick_i_final_suffix(self) -> str:
        """Choose which view's ``final_snapshot.png`` to use as ``I_final``."""
        if self.random_i_final_view:
            return self._rng.choice(VIEW_SUFFIXES)
        return ANCHOR_VIEW_SUFFIX

    def _load_i_final(self, part_id: str) -> Tuple[torch.Tensor, str]:
        """Return ``(I_final_tensor, view_suffix_used)``."""
        suffix = self._pick_i_final_suffix()
        path = os.path.join(self.data_root, f"{part_id}_{suffix}", I_FINAL_FILENAME)
        if not os.path.isfile(path):
            # Fall back to anchor view; if that's missing too, error out.
            fallback = os.path.join(
                self.data_root, f"{part_id}_{ANCHOR_VIEW_SUFFIX}", I_FINAL_FILENAME,
            )
            if os.path.isfile(fallback):
                logger.warning(
                    "Missing I_final for part=%s view=%s; falling back to anchor view.",
                    part_id, suffix,
                )
                suffix, path = ANCHOR_VIEW_SUFFIX, fallback
            else:
                raise FileNotFoundError(
                    f"Could not find {I_FINAL_FILENAME} for part={part_id} "
                    f"in either view {suffix} or anchor view {ANCHOR_VIEW_SUFFIX}."
                )
        img = Image.open(path).convert("RGB")
        return self._ifinal_tx(img), suffix

    def _load_prompt(self, part_id: str, roll_back_index: int) -> str:
        """Read the step prompt from the anchor view's folder, or fall back."""
        path = os.path.join(
            self._step_dir(part_id, ANCHOR_VIEW_SUFFIX, roll_back_index),
            PROMPT_FILENAME,
        )
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fp:
                text = fp.read().strip()
            if text:
                return text
        return self.fallback_prompt_template.format(
            part=part_id, step=roll_back_index,
        )

    # =========================================================== Dataset API
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.index[idx]

        # I_final --------------------------------------------------------
        i_final, i_final_view = self._load_i_final(rec.part_id)

        # G_target -------------------------------------------------------
        g_target = self._load_step_grid(rec.part_id, rec.cur_index)

        # G_prev ---------------------------------------------------------
        if rec.prev_index is None:
            g_prev = self._zero_grid()
        else:
            g_prev = self._load_step_grid(rec.part_id, rec.prev_index)

        # Prompt ---------------------------------------------------------
        prompt = self._load_prompt(rec.part_id, rec.cur_index)

        return {
            "I_final": i_final,
            "G_prev": g_prev,
            "G_target": g_target,
            "prompt": prompt,
            # Debug / logging metadata.
            "part_id": rec.part_id,
            "sorted_pos": rec.sorted_pos,
            "cur_index": rec.cur_index,
            "prev_index": rec.prev_index if rec.prev_index is not None else -1,
            "i_final_view": i_final_view,
        }


# ===========================================================================
def collate_cad_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack tensors and keep prompts / ids as plain Python lists."""
    out: Dict[str, object] = {
        "I_final":  torch.stack([b["I_final"]  for b in batch], dim=0),
        "G_prev":   torch.stack([b["G_prev"]   for b in batch], dim=0),
        "G_target": torch.stack([b["G_target"] for b in batch], dim=0),
        "prompt":       [b["prompt"]        for b in batch],   # List[str]
        "part_id":      [b["part_id"]       for b in batch],   # List[str]
        "sorted_pos":   [b["sorted_pos"]    for b in batch],   # List[int]
        "cur_index":    [b["cur_index"]     for b in batch],   # List[int]
        "prev_index":   [b["prev_index"]    for b in batch],   # List[int]
        "i_final_view": [b["i_final_view"]  for b in batch],   # List[str]
    }
    return out
