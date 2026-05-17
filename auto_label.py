"""Pseudo-label generator for CAD modeling steps using Qwen2.5-VL.

For every CAD part under ``--data-root``, this script walks the canonical
view folder (default ``<part>_PPP``), reads the **4-in-1 overlay**
``overlayed_all.png`` of each ``roll_back_index_N`` step, optionally loads
``operation_param.json`` as authoritative context, feeds them to
**Qwen2.5-VL**, and writes the resulting one-line description to
``prompt.txt`` inside that step folder.

The dataset has been simplified: the original 4 component PNGs
(``prev_depth_map.png``, ``sketch_plane_mask.png``, ``reference_mask.png``,
``result_frame.png``) are now pre-composited into a single ``overlayed_all.png``
by the data-prep stage, so this script no longer builds a 2 x 2 collage --
it just opens the overlay and forwards it to Qwen2.5-VL.

Color coding INSIDE the overlay (carried over from the wireframe layer of
``result_frame.png``):

    Red     -- the reference 2D sketch used by this step
    Green   -- edges of the newly ADDED solid entity
    Magenta -- edges of the REMOVED / CUT entity
    Blue    -- the termination face of the operation

Best-view selection still uses the depth-map differencing of
``prev_depth_map.png`` vs ``current_depth_map.png`` -- those files are kept
in every view folder for exactly this purpose.

The output location matches :data:`config.PROMPT_FILENAME`, which is where
:class:`dataset.CADMultiViewDataset._load_prompt` already looks for prompts,
so no further wiring is needed once labels are generated.

Usage
-----
::

    pip install -U transformers>=4.49.0 qwen-vl-utils>=0.0.10
    python auto_label.py --data-root ./data
    # Optional flags:
    #   --model Qwen/Qwen2.5-VL-3B-Instruct   # smaller, faster
    #   --view-selection auto|fixed            # pick best view per step (default: auto)
    #   --view-suffix PPP                      # fallback view in auto / forced view in fixed
    #   --no-operation-params                  # disable JSON grounding (default: ON)
    #   --overwrite                            # re-label even if prompt.txt exists
    #   --broadcast                            # copy prompt.txt to all 8 views
    #   --include-final-snapshot               # add part's final image as extra context
    #   --max-parts 10                         # quick smoke test
    #   --dry-run                              # print plan, don't load the MLLM
    #   --debug-overlay-dir ./out/overlays     # save a copy of every input image

Resume / safety
---------------
The script is **resume-safe**: any step whose ``prompt.txt`` already exists
and is non-empty is skipped unless ``--overwrite`` is passed. Generation
failures are logged and counted but do not abort the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from config import (
    ANCHOR_VIEW_SUFFIX,
    CURRENT_DEPTH_FILENAME,
    I_FINAL_FILENAME,
    OPERATION_PARAM_FILENAME,
    OVERLAYED_FILENAME,
    PRETRAINED_DIR,
    PREV_DEPTH_FILENAME,
    PROMPT_FILENAME,
    STEP_DIR_PREFIX,
    VIEW_SUFFIXES,
)


# ===========================================================================
# Static prompts
# ===========================================================================

SYSTEM_PROMPT = """You are an expert in CAD reverse engineering. I will provide a SINGLE composite image representing one incremental step in a CAD modeling sequence. The image is a 4-in-1 overlay that combines, on the same canvas:
- A grayscale base representing the depth map of the part BEFORE this operation.
- A semi-transparent YELLOW mask indicating the sketch plane used for this operation.
- A semi-transparent CYAN mask indicating reference geometry/paths (if any).
- A LOCAL feature wireframe showing ONLY the local entity created, modified, or removed in this exact step.

CRITICAL WIREFRAME COLOR CODING:
- Red: The reference 2D sketch used by this operation.
- Green: Edges of the newly ADDED solid local entity.
- Magenta: Edges of the REMOVED / CUT local entity.
- Blue: The local entity termination face of this operation.

GROUND-TRUTH OPERATION PARAMETERS (JSON):
You will also receive a JSON block containing the true parameters for this operation (e.g., modeling_type, construct_type, is_symmetric). Use this JSON strictly to determine the topological operation intent (whether it is an addition or a cut, what the specific tool name is). 
DO NOT output any specific numerical dimensions (such as depth=10 or radius=5). The geometric scale and proportions are implicitly embedded in the image and do not need to be quantified in text.

Analyze the visual inputs and the JSON, then write a single, concise sentence describing the operation. Format MUST be: '[Operation Type]: Based on [sketch shape] on the [sketch plane/reference], generated [entity changes].'

For example:
- 'Extrude: Based on the red circular sketch drawn on the yellow sketch plane, extruded a green solid cylinder up to the blue termination face.'
- 'Sweep: Based on the red rectangular sketch on the yellow sketch plane and guided by the cyan reference path, generated a green solid sweep feature.'
- 'Extruded Cut: Based on the red hexagonal sketch on the yellow sketch plane, cut a magenta negative space up to the blue termination face.'"""

# Minimal user-side payload text. The system prompt already specifies the
# layout, color coding, and required output format -- here we just nudge
# the model to actually produce that single sentence.
USER_INSTRUCTION = (
    "Analyze the overlay image above and respond with ONE sentence "
    "in the required format. No preamble, no markdown, no extra commentary."
)

_STEP_RE = re.compile(rf"^{re.escape(STEP_DIR_PREFIX)}(\d+)$")

# Post-processing: strip stray markdown / preamble that some MLLM checkpoints
# emit despite the explicit instruction. We deliberately do NOT strip a
# leading "<Word>:" because that's the required "[Operation Type]:" prefix.
_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*(here(?:'s| is)?|sure[,!]?|certainly[,!]?|of course[,!]?)\b[^.]*[.!?:]\s*",
               re.IGNORECASE),
    re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*description\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*response\s*:\s*", re.IGNORECASE),
]


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class LabelerConfig:
    """Everything the labeler script needs in one place."""

    data_root: str
    view_suffix: str = ANCHOR_VIEW_SUFFIX
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    # The new format is more verbose ("Extrusion: Based on ..., generated ...");
    # 96 tokens leaves headroom while still capping cost.
    max_new_tokens: int = 96
    # Qwen processor "dynamic resolution" knobs. The 4-in-1 overlay is a
    # single image whose native size depends on the data-prep stage. We
    # leave plenty of room (max_pixels ~1.1 MP) to avoid downscaling
    # high-resolution overlays.
    min_pixels: int = 256 * 28 * 28        # ~200 k pixels
    max_pixels: int = 1408 * 28 * 28       # ~1.10 M pixels
    overwrite: bool = False
    broadcast_to_all_views: bool = False
    # Default OFF: the SYSTEM_PROMPT is structured around the SINGLE overlay
    # image; adding a 2nd image would contradict it. Pass
    # ``--include-final-snapshot`` to attach it as supplementary context.
    include_final_snapshot: bool = False
    max_parts: Optional[int] = None
    log_every: int = 25
    use_flash_attention: bool = True
    dry_run: bool = False
    # Optional: copy every overlay we send to Qwen into this folder (under
    # a deterministic name) for offline visual debugging.
    debug_overlay_dir: Optional[str] = None

    # ---- Dynamic best-view selection ----
    # "auto"  -> use depth-map differencing to pick the least-occluded view
    #            per (part, step). ``view_suffix`` is then used only as the
    #            FALLBACK view when no view has usable depth maps.
    # "fixed" -> always use ``view_suffix`` (old behaviour).
    view_selection_mode: str = "auto"
    view_diff_threshold: float = 0.01      # normalized [0,1] depth difference
    min_visible_pixels: int = 1            # minimum delta-pixel count to accept a view
    save_view_selections: Optional[str] = None  # optional JSON path for QA trail

    # ---- Ground-truth operation parameters ----
    # When True (default), read ``operation_param.json`` from the anchor
    # view's step folder and feed it to Qwen2.5-VL as authoritative text.
    include_operation_params: bool = True
    operation_params_max_chars: int = 4096


# ===========================================================================
# Qwen2.5-VL wrapper
# ===========================================================================
class Qwen25VLLabeler:
    """Thin wrapper around the HF Qwen2.5-VL model.

    Loads model + processor once, then exposes :meth:`describe_step` which
    composes a 2 x 2 collage of the 4 step components and returns a single
    cleaned-up string describing the modeling operation.
    """

    def __init__(self, cfg: LabelerConfig) -> None:
        self.cfg = cfg

        # Lazy imports so ``--dry-run`` doesn't need the heavy deps installed.
        import torch
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "Cannot import Qwen2.5-VL classes from transformers. "
                "Please install: pip install -U 'transformers>=4.49.0'"
            ) from e
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise RuntimeError(
                "qwen_vl_utils not installed. Please install: "
                "pip install -U 'qwen-vl-utils>=0.0.10'"
            ) from e

        self._torch = torch
        self._process_vision_info = process_vision_info

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[cfg.dtype]

        load_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map="auto" if cfg.device.startswith("cuda") else None,
        )
        if cfg.use_flash_attention and cfg.device.startswith("cuda"):
            # Falls back silently if FA2 is unavailable.
            load_kwargs["attn_implementation"] = "flash_attention_2"

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
            cache_dir=PRETRAINED_DIR,
        )

        # Cache redirection for the model snapshot itself.
        load_kwargs["cache_dir"] = PRETRAINED_DIR

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_name_or_path, **load_kwargs
            ).eval()
        except Exception:
            # Retry without flash-attention if its install is broken.
            load_kwargs.pop("attn_implementation", None)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_name_or_path, **load_kwargs
            ).eval()

        if not cfg.device.startswith("cuda"):
            self.model = self.model.to(cfg.device)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_overlay_image(path: Optional[str]):
        """Load ``overlayed_all.png`` (or ``None``) into a PIL ``RGB`` image.

        When ``path`` is ``None`` or missing we synthesise a tiny black
        placeholder so the model still sees *something* and we can log a
        warning rather than crash mid-batch.
        """
        from PIL import Image  # lazy: keep --dry-run light
        if path is not None and os.path.isfile(path):
            return Image.open(path).convert("RGB")
        return Image.new("RGB", (256, 256), color=(0, 0, 0))

    @staticmethod
    def _file_uri(abs_path: str) -> str:
        """Build a ``file://`` URI that qwen-vl-utils understands on any OS."""
        path = os.path.abspath(abs_path).replace("\\", "/")
        if not path.startswith("/"):
            path = "/" + path                # Windows: file:///C:/...
        return f"file://{path}"

    @staticmethod
    def _postprocess(text: str) -> str:
        """Trim whitespace, drop preambles / markdown, but keep '<Op>:' prefix."""
        s = text.strip()
        # Strip wrapping quotes if the model added any.
        for q in ("\"", "'", "“", "”", "‘", "’"):
            if s.startswith(q) and s.endswith(q):
                s = s[1:-1].strip()
        # Strip leading **bold** wrapper around the whole sentence.
        if s.startswith("**") and s.endswith("**"):
            s = s[2:-2].strip()
        # Drop leading bullets / list numbering. NOTE: we stop at the FIRST
        # non-bullet character so we don't eat into "Extrusion:" or "Cut:".
        s = re.sub(r"^[-*\u2022]+\s*", "", s)
        s = re.sub(r"^\d+[.)]\s+", "", s)
        # Strip well-known preamble phrases (these patterns explicitly do NOT
        # match a leading "<OperationType>:" since they only consume the words
        # listed in the regex alternation).
        for pattern in _PREAMBLE_PATTERNS:
            s = pattern.sub("", s)
        # Collapse internal whitespace.
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ------------------------------------------------------------------ inference
    def describe_step(
        self,
        overlay_path: Optional[str],
        final_snapshot_path: Optional[str] = None,
        debug_overlay_save_path: Optional[str] = None,
        params_json_text: Optional[str] = None,
    ) -> str:
        """Run one MLLM forward pass for a single modeling step.

        Parameters
        ----------
        overlay_path:
            Absolute path to the step's ``overlayed_all.png`` (the 4-in-1
            composite produced by data prep). ``None`` => fall back to a
            black placeholder; the call will still complete but the result
            is meaningless. The caller is expected to log this case.
        final_snapshot_path:
            Optional path to the part's ``final_snapshot.png``. If provided
            it is shown AFTER the overlay as supplementary context with an
            explicit text label so the model does not confuse it with the
            primary input. **Off by default**.
        debug_overlay_save_path:
            If set, copy the overlay we send to Qwen to this path. Useful
            for eyeballing what the model actually sees, especially when
            best-view selection picked a non-PPP view.
        params_json_text:
            Optional pretty-printed JSON string of ground-truth operation
            parameters. Injected into the user turn as an authoritative
            text block right before the final instruction, wrapped in a
            fenced markdown code block so the model can parse it cleanly.

        Returns
        -------
        str
            A cleaned-up single-sentence description in the format
            ``"<Operation>: Based on <sketch/ref>, generated <changes>."``.
        """
        torch = self._torch

        # ---- 1) Load the overlay PNG ----
        overlay = self._load_overlay_image(overlay_path)
        if debug_overlay_save_path is not None:
            parent = os.path.dirname(debug_overlay_save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            overlay.save(debug_overlay_save_path)

        # ---- 2) Compose the chat messages ----
        # qwen-vl-utils handles PIL images in the "image" field directly.
        user_content: List[dict] = [
            {"type": "image", "image": overlay},
        ]
        if final_snapshot_path is not None and os.path.isfile(final_snapshot_path):
            user_content.append({
                "type": "text",
                "text": (
                    "Supplementary global context: the part's final shape "
                    "(NOT part of the overlay above)."
                ),
            })
            user_content.append({"type": "image", "image": self._file_uri(final_snapshot_path)})

        # Inject ground-truth operation parameters AFTER the image(s) but
        # BEFORE the final user instruction. The system prompt already
        # designates this JSON as authoritative.
        if params_json_text:
            user_content.append({
                "type": "text",
                "text": (
                    "GROUND-TRUTH OPERATION PARAMETERS (authoritative; "
                    "use to ground your description):\n"
                    "```json\n" + params_json_text + "\n```"
                ),
            })

        user_content.append({"type": "text", "text": USER_INSTRUCTION})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # ---- 3) Tokenize + pack vision inputs ----
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # ---- 4) Greedy decode ----
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )

        # Slice off prompt tokens, decode, post-process.
        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._postprocess(raw)


# ===========================================================================
# Dynamic best-view selection via depth-map differencing
# ===========================================================================
#
# Why this exists
# ---------------
# A CAD operation can be entirely *occluded* by existing geometry in some
# views. Labeling such a step from a fixed view (e.g. PPP) feeds Qwen2.5-VL
# images where the new feature is invisible -> poor labels.
#
# Approach: for each of the 8 views, compare the BEFORE depth map
# (``prev_depth_map.png``) to the AFTER depth map (``current_depth_map.png``)
# pixel-by-pixel. Pixels whose depth changed represent the visible
# silhouette of the new feature in that view. The view with the most
# changed pixels is the least occluded -- pick it for labeling.


def _load_normalized_depth(path: str) -> "Optional[object]":
    """Load a depth map and return a ``[0, 1]`` float32 numpy array.

    Handles the common PIL depth modes:
      * 8-bit ``L`` / ``RGB`` -> divide by 255
      * 16-bit ``I`` / ``I;16`` -> divide by per-image max (or 65535 if all-zero)

    Returns ``None`` if the file is missing or unreadable.
    """
    import numpy as np  # lazy: keeps ``--dry-run`` light
    from PIL import Image

    if not os.path.isfile(path):
        return None
    try:
        img = Image.open(path)
        if img.mode in ("I", "I;16", "I;16B", "I;16L"):
            arr = np.asarray(img, dtype=np.float32)
            denom = float(arr.max()) if float(arr.max()) > 0 else 65535.0
            return arr / denom
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        return arr / 255.0
    except Exception:
        return None


def score_view_visibility(
    data_root: str,
    part_id: str,
    view_suffix: str,
    roll_back_index: int,
    threshold: float = 0.01,
) -> Optional[int]:
    """Count pixels whose depth changed beyond ``threshold`` in this view.

    Returns
    -------
    int  -- number of changed pixels (higher == feature is more visible).
    None -- one or both depth maps missing / shapes don't match.
    """
    import numpy as np

    step_dir = os.path.join(
        data_root,
        f"{part_id}_{view_suffix}",
        f"{STEP_DIR_PREFIX}{roll_back_index}",
    )
    prev = _load_normalized_depth(os.path.join(step_dir, PREV_DEPTH_FILENAME))
    curr = _load_normalized_depth(os.path.join(step_dir, CURRENT_DEPTH_FILENAME))
    if prev is None or curr is None:
        return None
    if prev.shape != curr.shape:
        return None
    diff = np.abs(curr - prev)
    return int((diff > threshold).sum())


@dataclass(frozen=True)
class ViewSelection:
    """Outcome of best-view selection for one ``(part, step)`` pair."""

    selected_view: str
    score: int
    all_scores: Dict[str, Optional[int]]
    fell_back: bool                 # True if no view had usable depth data
    reason: str = ""                # human-readable explanation


def select_best_view(
    data_root: str,
    part_id: str,
    roll_back_index: int,
    candidate_views: Tuple[str, ...] = VIEW_SUFFIXES,
    threshold: float = 0.01,
    min_visible_pixels: int = 1,
    default_view: str = ANCHOR_VIEW_SUFFIX,
) -> ViewSelection:
    """Pick the view where the new feature is most visible.

    Ties broken by ``candidate_views`` order (earlier wins).
    Falls back to ``default_view`` if no view meets ``min_visible_pixels``.
    """
    scores: Dict[str, Optional[int]] = {
        v: score_view_visibility(data_root, part_id, v, roll_back_index, threshold)
        for v in candidate_views
    }

    valid = [(v, s) for v, s in scores.items()
             if s is not None and s >= min_visible_pixels]

    if not valid:
        if all(s is None for s in scores.values()):
            reason = "no depth maps available in any view"
        else:
            reason = f"no view reached min_visible_pixels={min_visible_pixels}"
        return ViewSelection(
            selected_view=default_view,
            score=int(scores.get(default_view) or 0),
            all_scores=scores,
            fell_back=True,
            reason=reason,
        )

    best_view, best_score = max(
        valid,
        key=lambda kv: (kv[1], -candidate_views.index(kv[0])),
    )
    return ViewSelection(
        selected_view=best_view,
        score=int(best_score),
        all_scores=scores,
        fell_back=False,
    )


# ===========================================================================
# Ground-truth operation parameters loader
# ===========================================================================
def load_operation_params(
    data_root: str,
    part_id: str,
    roll_back_index: int,
    anchor_view: str = ANCHOR_VIEW_SUFFIX,
    max_chars: int = 4096,
) -> Optional[str]:
    """Read ``operation_param.json`` from the anchor view's step folder and
    return it pretty-printed as a string ready to embed in a chat message.

    The JSON is view-invariant by definition (parametric data describing the
    operation), so we only ever read from the anchor view.

    Parameters
    ----------
    max_chars:
        Hard cap on the pretty-printed length. If exceeded, the output is
        truncated and a clear "...[truncated; ...]" marker is appended so
        the MLLM is aware. Default 4096 chars (~1k tokens) keeps even a
        verbose schema well under the Qwen2.5-VL context budget.

    Returns
    -------
    str  -- pretty-printed JSON text (already including any truncation marker).
    None -- file missing, unreadable, or malformed JSON.
    """
    json_path = os.path.join(
        data_root,
        f"{part_id}_{anchor_view}",
        f"{STEP_DIR_PREFIX}{roll_back_index}",
        OPERATION_PARAM_FILENAME,
    )
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger("auto_label").warning(
            "Failed to read %s: %s", json_path, exc,
        )
        return None

    # ``sort_keys=False`` to preserve whatever ordering the data prep wrote
    # (often semantically meaningful: e.g. type first, then params).
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if len(text) > max_chars:
        head = text[: max_chars - 64]
        suffix = (
            f"\n  ...[truncated; original {len(text)} chars, "
            f"showing first {max_chars - 64}]"
        )
        text = head + suffix
    return text


# ===========================================================================
# Dataset walking helpers
# ===========================================================================
def discover_part_ids(data_root: str, view_suffix: str) -> List[str]:
    """Return all ``CAD_PART_ID`` values that have a ``<id>_<view_suffix>`` folder."""
    suffix = f"_{view_suffix}"
    out: List[str] = []
    for name in sorted(os.listdir(data_root)):
        if name.endswith(suffix) and os.path.isdir(os.path.join(data_root, name)):
            out.append(name[: -len(suffix)])
    return out


def discover_step_indices(view_dir: str) -> List[int]:
    """Return the sorted roll-back indices inside one view folder."""
    if not os.path.isdir(view_dir):
        return []
    indices: List[int] = []
    for entry in os.listdir(view_dir):
        m = _STEP_RE.match(entry)
        if m and os.path.isdir(os.path.join(view_dir, entry)):
            indices.append(int(m.group(1)))
    indices.sort()
    return indices


def collect_step_overlay(step_dir: str) -> Optional[str]:
    """Return absolute path to ``overlayed_all.png`` in this step folder, or ``None``."""
    path = os.path.join(step_dir, OVERLAYED_FILENAME)
    return os.path.abspath(path) if os.path.isfile(path) else None


def maybe_broadcast(
    data_root: str,
    part_id: str,
    source_suffix: str,
    roll_back_index: int,
    description: str,
    overwrite: bool,
) -> int:
    """Copy ``description`` to every other view folder's prompt.txt.

    Returns the number of files written.
    """
    written = 0
    for suffix in VIEW_SUFFIXES:
        if suffix == source_suffix:
            continue
        other_dir = os.path.join(
            data_root, f"{part_id}_{suffix}",
            f"{STEP_DIR_PREFIX}{roll_back_index}",
        )
        if not os.path.isdir(other_dir):
            # Some views may not exist or this step may be missing there.
            continue
        target = os.path.join(other_dir, PROMPT_FILENAME)
        if os.path.isfile(target) and os.path.getsize(target) > 0 and not overwrite:
            continue
        with open(target, "w", encoding="utf-8") as fp:
            fp.write(description + "\n")
        written += 1
    return written


# ===========================================================================
# Main loop
# ===========================================================================
def run(cfg: LabelerConfig) -> None:
    logger = logging.getLogger("auto_label")

    if cfg.view_suffix not in VIEW_SUFFIXES:
        raise ValueError(
            f"view_suffix='{cfg.view_suffix}' is not one of {VIEW_SUFFIXES}."
        )
    if cfg.view_selection_mode not in ("auto", "fixed"):
        raise ValueError(
            f"view_selection_mode must be 'auto' or 'fixed', got "
            f"{cfg.view_selection_mode!r}."
        )
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"data_root not found: {cfg.data_root}")

    # In auto mode we enumerate parts/steps via the anchor view (PPP),
    # guaranteed to exist by the dataset contract. The actual analysis view
    # is chosen per step via depth differencing.
    discovery_view = ANCHOR_VIEW_SUFFIX if cfg.view_selection_mode == "auto" else cfg.view_suffix
    part_ids = discover_part_ids(cfg.data_root, discovery_view)
    if cfg.max_parts is not None:
        part_ids = part_ids[: cfg.max_parts]
    logger.info(
        "Found %d parts (enumerated via '%s'). View selection: %s. "
        "Operation params: %s.",
        len(part_ids), discovery_view, cfg.view_selection_mode,
        "ON" if cfg.include_operation_params else "OFF",
    )
    if not part_ids:
        logger.warning("Nothing to do. Exiting.")
        return

    # Build a plan first so --dry-run never instantiates the model.
    # Each entry: (part_id, roll_back_index, anchor_step_dir_for_prompt_save).
    plan: List[Tuple[str, int, str]] = []
    for part_id in part_ids:
        anchor_view_dir = os.path.join(cfg.data_root, f"{part_id}_{discovery_view}")
        for idx in discover_step_indices(anchor_view_dir):
            anchor_step_dir = os.path.join(anchor_view_dir, f"{STEP_DIR_PREFIX}{idx}")
            plan.append((part_id, idx, anchor_step_dir))

    logger.info("Planned %d (part, step) pairs.", len(plan))
    if cfg.dry_run:
        for part_id, idx, step_dir in plan[:20]:
            if cfg.view_selection_mode == "auto":
                sel = select_best_view(
                    cfg.data_root, part_id, idx,
                    threshold=cfg.view_diff_threshold,
                    min_visible_pixels=cfg.min_visible_pixels,
                )
                tag = f"selected={sel.selected_view}({sel.score} px)"
                if sel.fell_back:
                    tag += f"  FALLBACK [{sel.reason}]"
            else:
                tag = f"fixed={cfg.view_suffix}"
            params_tag = ""
            if cfg.include_operation_params:
                p = load_operation_params(
                    cfg.data_root, part_id, idx,
                    max_chars=cfg.operation_params_max_chars,
                )
                params_tag = f"  params={'yes' if p else 'MISSING'}"
            logger.info(
                "DRY-RUN would label: %s step=%d  %s%s",
                step_dir, idx, tag, params_tag,
            )
        if len(plan) > 20:
            logger.info("... and %d more.", len(plan) - 20)
        return

    labeler = Qwen25VLLabeler(cfg)

    total = 0
    written = 0
    skipped = 0
    broadcast = 0
    failures = 0
    fell_back = 0
    missing_params = 0
    selection_log: List[Dict[str, object]] = []
    t0 = time.time()

    pbar = tqdm(plan, desc="auto-labeling", smoothing=0.05)
    for part_id, idx, anchor_step_dir in pbar:
        total += 1

        # Prompt always goes to the ANCHOR view's step folder so
        # ``dataset.CADMultiViewDataset._load_prompt`` can find it.
        prompt_path = os.path.join(anchor_step_dir, PROMPT_FILENAME)

        # Resume: skip steps already labeled (unless --overwrite).
        if (
            os.path.isfile(prompt_path)
            and os.path.getsize(prompt_path) > 0
            and not cfg.overwrite
        ):
            skipped += 1
            continue

        # ---- Pick the analysis view ----
        if cfg.view_selection_mode == "auto":
            sel = select_best_view(
                cfg.data_root, part_id, idx,
                threshold=cfg.view_diff_threshold,
                min_visible_pixels=cfg.min_visible_pixels,
                default_view=cfg.view_suffix,
            )
            analysis_view = sel.selected_view
            if sel.fell_back:
                fell_back += 1
                logger.warning(
                    "Best-view selection fell back to '%s' for %s/step_%d (%s).",
                    analysis_view, part_id, idx, sel.reason,
                )
            if cfg.save_view_selections:
                selection_log.append({
                    "part_id":   part_id,
                    "step":      idx,
                    "selected":  analysis_view,
                    "score":     sel.score,
                    "all_scores": sel.all_scores,
                    "fell_back": sel.fell_back,
                    "reason":    sel.reason,
                })
        else:
            analysis_view = cfg.view_suffix

        analysis_step_dir = os.path.join(
            cfg.data_root, f"{part_id}_{analysis_view}", f"{STEP_DIR_PREFIX}{idx}",
        )
        overlay_path = collect_step_overlay(analysis_step_dir)
        if overlay_path is None:
            logger.warning(
                "Missing %s in %s -- skipping (model would only see a black image).",
                OVERLAYED_FILENAME, analysis_step_dir,
            )
            failures += 1
            continue

        # ---- Load ground-truth operation parameters (view-invariant) ----
        params_text: Optional[str] = None
        if cfg.include_operation_params:
            params_text = load_operation_params(
                cfg.data_root, part_id, idx,
                anchor_view=ANCHOR_VIEW_SUFFIX,
                max_chars=cfg.operation_params_max_chars,
            )
            if params_text is None:
                missing_params += 1

        # Optional global reference image (off by default).
        final_snapshot: Optional[str] = None
        if cfg.include_final_snapshot:
            cand = os.path.join(
                cfg.data_root, f"{part_id}_{analysis_view}", I_FINAL_FILENAME,
            )
            if os.path.isfile(cand):
                final_snapshot = cand

        debug_save: Optional[str] = None
        if cfg.debug_overlay_dir:
            debug_save = os.path.join(
                cfg.debug_overlay_dir,
                f"{part_id}_{analysis_view}__step_{idx:04d}.png",
            )

        try:
            description = labeler.describe_step(
                overlay_path=overlay_path,
                final_snapshot_path=final_snapshot,
                debug_overlay_save_path=debug_save,
                params_json_text=params_text,
            )
        except Exception as exc:
            logger.error(
                "MLLM call failed for %s/%s/step_%d: %s",
                part_id, analysis_view, idx, exc,
            )
            failures += 1
            continue

        if not description:
            logger.warning(
                "Empty description for %s/%s/step_%d; skipping.",
                part_id, analysis_view, idx,
            )
            failures += 1
            continue

        with open(prompt_path, "w", encoding="utf-8") as fp:
            fp.write(description + "\n")
        written += 1

        # Broadcast from the anchor view (which is where the prompt landed).
        if cfg.broadcast_to_all_views:
            broadcast += maybe_broadcast(
                cfg.data_root, part_id,
                source_suffix=ANCHOR_VIEW_SUFFIX,
                roll_back_index=idx,
                description=description,
                overwrite=cfg.overwrite,
            )

        if (written + failures) % cfg.log_every == 0:
            elapsed = time.time() - t0
            ips = (written + failures) / max(1.0, elapsed)
            pbar.set_postfix(
                wrote=written, skipped=skipped, failed=failures,
                broadcast=broadcast, fb=fell_back, no_params=missing_params,
                last_view=analysis_view, ips=f"{ips:.2f}",
            )

    if cfg.save_view_selections and selection_log:
        out_path = cfg.save_view_selections
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(selection_log, fp, indent=2)
        logger.info(
            "Wrote %d view-selection records to %s.",
            len(selection_log), out_path,
        )

    logger.info(
        "Done. %d total, %d written, %d skipped, %d broadcast, "
        "%d view-fallbacks, %d missing-params, %d failed. (%.1fs)",
        total, written, skipped, broadcast,
        fell_back, missing_params, failures,
        time.time() - t0,
    )


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args() -> LabelerConfig:
    p = argparse.ArgumentParser(
        description="Auto-generate prompt.txt for CAD modeling steps with Qwen2.5-VL.",
    )
    p.add_argument("--data-root", type=str, required=True,
                   help="Root folder containing the <part>_<SUFFIX> directories.")
    p.add_argument("--view-suffix", type=str, default=ANCHOR_VIEW_SUFFIX,
                   choices=list(VIEW_SUFFIXES),
                   help="In --view-selection=fixed: the view to read images from. "
                        "In --view-selection=auto: the FALLBACK view when no view "
                        "has usable depth maps (default: PPP).")
    p.add_argument("--view-selection", type=str, default="auto",
                   choices=["auto", "fixed"],
                   help="'auto' (default): pick the least-occluded view per step "
                        "via depth-map differencing. 'fixed': always use --view-suffix.")
    p.add_argument("--view-diff-threshold", type=float, default=0.01,
                   help="Normalized [0,1] depth difference required to count a "
                        "pixel as 'changed' when scoring view visibility. (default: 0.01)")
    p.add_argument("--min-visible-pixels", type=int, default=1,
                   help="Minimum delta-pixel count a view needs to be considered. "
                        "If no view meets this bar, falls back to --view-suffix.")
    p.add_argument("--save-view-selections", type=str, default=None,
                   help="If set, write a JSON file of per-step view selections (for QA).")
    p.add_argument("--no-operation-params", action="store_true",
                   help="Disable feeding operation_param.json as authoritative "
                        "ground-truth context. (Default: ON.)")
    p.add_argument("--operation-params-max-chars", type=int, default=4096,
                   help="Truncate the rendered JSON beyond this many characters. "
                        "(default: 4096)")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HuggingFace id or local path of the Qwen2.5-VL checkpoint.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda | cuda:0 | cpu | mps")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=96,
                   help="Max generated tokens (the new format is more verbose).")
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=1408 * 28 * 28,
                   help="Upper bound on collage pixel count after Qwen smart-resize.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if prompt.txt already exists.")
    p.add_argument("--broadcast", action="store_true",
                   help="Copy each generated prompt.txt to the other 7 view folders.")
    p.add_argument("--include-final-snapshot", action="store_false",
                   help="Append the part's final_snapshot.png as supplementary "
                        "context AFTER the 2x2 collage. Off by default.")
    p.add_argument("--no-flash-attn", action="store_false",
                   help="Disable flash-attention-2 (use vanilla SDPA).")
    p.add_argument("--max-parts", type=int, default=None,
                   help="Stop after this many parts (smoke testing).")
    p.add_argument("--debug-collage-dir", type=str, default=None,
                   help="If set, save every collage as PNG here for visual inspection.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit; do not load the MLLM.")
    args = p.parse_args()

    return LabelerConfig(
        data_root=args.data_root,
        view_suffix=args.view_suffix,
        model_name_or_path=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        overwrite=args.overwrite,
        broadcast_to_all_views=args.broadcast,
        include_final_snapshot=args.include_final_snapshot,
        use_flash_attention=not args.no_flash_attn,
        max_parts=args.max_parts,
        dry_run=args.dry_run,
        view_selection_mode=args.view_selection,
        view_diff_threshold=args.view_diff_threshold,
        min_visible_pixels=args.min_visible_pixels,
        save_view_selections=args.save_view_selections,
        include_operation_params=not args.no_operation_params,
        operation_params_max_chars=args.operation_params_max_chars,
    )


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    cfg = _parse_args()
    try:
        run(cfg)
    except KeyboardInterrupt:
        logging.getLogger("auto_label").info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
