"""Pseudo-label generator for CAD modeling steps using Qwen2.5-VL.

For every CAD part under ``--data-root``, this script walks the canonical
view folder (default ``<part>_PPP``), reads the 4 component PNGs of each
``roll_back_index_N`` step, **composes them into a single 2 x 2 collage**
matching the layout described in :data:`SYSTEM_PROMPT`, feeds the collage
to **Qwen2.5-VL**, and writes the resulting one-line description to
``prompt.txt`` inside that step folder.

Critical: ``result_frame.png`` (bottom-right cell of the collage) shows the
**LOCAL feature delta** of the current step only -- not the wireframe of
the entire part. Color coding inside that image:

    Red     -- the reference 2D sketch used by this step
    Green   -- edges of the newly ADDED solid entity
    Magenta -- edges of the REMOVED / CUT entity
    Blue    -- the termination face of the operation

The output location matches :data:`config.PROMPT_FILENAME`, which is where
:class:`dataset.CADMultiViewDataset._load_prompt` already looks for prompts,
so no further wiring is needed once labels are generated.

Usage
-----
::

    # 1.   Install model deps (one-time):
    #        pip install -U transformers>=4.49.0 qwen-vl-utils>=0.0.10
    # 2.   Run the labeler:
    python auto_label.py --data-root ./data
    # Optional flags:
    #   --model Qwen/Qwen2.5-VL-3B-Instruct   # smaller, faster
    #   --view-suffix PPP                      # which canonical view to read
    #   --overwrite                            # re-label even if prompt.txt exists
    #   --broadcast                            # copy prompt.txt to all 8 views
    #   --include-final-snapshot               # add part's final image as extra context
    #   --max-parts 10                         # quick smoke test
    #   --dry-run                              # print plan, don't load the MLLM
    #   --debug-collage-dir ./out/collages     # dump every collage as PNG for inspection

Resume / safety
---------------
The script is **resume-safe**: any step whose ``prompt.txt`` already exists
and is non-empty is skipped unless ``--overwrite`` is passed. Generation
failures are logged and counted but do not abort the run.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tqdm.auto import tqdm

from config import (
    ANCHOR_VIEW_SUFFIX,
    I_FINAL_FILENAME,
    PROMPT_FILENAME,
    ROW_FILENAMES,
    STEP_DIR_PREFIX,
    VIEW_SUFFIXES,
)


# ===========================================================================
# Static prompts
# ===========================================================================

# Position labels of the 2 x 2 collage cells. Order corresponds 1-to-1 with
# ``config.ROW_FILENAMES`` (i.e. the canonical row order used everywhere
# else in the project).
#   index 0 -> Top-Left      = prev_depth_map.png
#   index 1 -> Top-Right     = sketch_plane_mask.png
#   index 2 -> Bottom-Left   = reference_mask.png
#   index 3 -> Bottom-Right  = result_frame.png   (LOCAL feature wireframe)
COLLAGE_POSITIONS: Tuple[str, ...] = (
    "top-left", "top-right", "bottom-left", "bottom-right",
)

SYSTEM_PROMPT = """You are an expert in CAD reverse engineering. I will provide a 2x2 collaged image representing a SINGLE incremental step in a CAD modeling sequence. 
- Top-Left: Depth map of the entire model BEFORE this operation. 
- Top-Right: Mask indicating the sketch plane for this operation. 
- Bottom-Left: Mask indicating reference geometry (if any). 
- Bottom-Right: The LOCAL feature wireframe. 

CRITICAL INSTRUCTION FOR BOTTOM-RIGHT IMAGE:
This image is NOT the wireframe of the entire CAD part. It shows ONLY the specific entity created, modified, or removed in this exact step. 
Color coding for this local feature:
- Red: The reference sketch used for this specific operation.
- Green: Edges of the newly ADDED solid entity in this step.
- Magenta: Edges of the REMOVED/CUT entity in this step.
- Blue: The termination face of this specific operation.

Analyze the images and write a single, concise sentence describing the operation. Format MUST be: '[Operation Type]: Based on [sketch/reference], generated [entity changes].' 
For example: 'Extrusion: Based on the red circular sketch, extruded a green solid cylinder up to the blue termination face.'"""

# Minimal user-side payload text. The system prompt already specifies the
# layout, color coding, and required output format -- here we just nudge
# the model to actually produce that single sentence.
USER_INSTRUCTION = (
    "Analyze the 2x2 collaged image above and respond with ONE sentence "
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
    # Qwen processor "dynamic resolution" knobs. The single 2 x 2 collage
    # is ~1 MP at default cell_size=512, so max_pixels must >= ~1.05 MP.
    min_pixels: int = 256 * 28 * 28        # ~200 k pixels
    max_pixels: int = 1408 * 28 * 28       # ~1.10 M pixels (fits a 1024x1024 collage)
    # Per-cell pixel size of the 2 x 2 collage. Final collage = (2*cs, 2*cs).
    collage_cell_size: int = 512
    overwrite: bool = False
    broadcast_to_all_views: bool = False
    # Default OFF: the new SYSTEM_PROMPT is structured around the SINGLE
    # 2 x 2 collage; adding a 5th image would contradict it. Pass
    # ``--include-final-snapshot`` to attach it as supplementary context.
    include_final_snapshot: bool = False
    max_parts: Optional[int] = None
    log_every: int = 25
    use_flash_attention: bool = True
    dry_run: bool = False
    # Optional: dump every collage as PNG to this folder for visual debugging.
    debug_collage_dir: Optional[str] = None


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
        )

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
    def _build_collage_image(
        component_paths: List[Optional[str]],
        cell_size: int = 512,
    ) -> "object":
        """Compose 4 component PNGs into a single 2 x 2 collage PIL image.

        Layout (must match :data:`SYSTEM_PROMPT` literally!)::

            +-------------------+-------------------+
            | TL: prev_depth    | TR: sketch_plane  |
            |     _map.png      |     _mask.png     |
            +-------------------+-------------------+
            | BL: reference     | BR: result_frame  |
            |     _mask.png     |     .png (LOCAL)  |
            +-------------------+-------------------+

        Missing files (``None`` entries) become solid black cells so the
        spatial layout the system prompt references is always preserved.
        """
        from PIL import Image  # imported lazily so --dry-run stays light

        if len(component_paths) != 4:
            raise ValueError(
                f"_build_collage_image expects exactly 4 paths, got {len(component_paths)}."
            )

        cells = []
        for path in component_paths:
            if path is not None and os.path.isfile(path):
                img = Image.open(path).convert("RGB")
                # Resize to the canonical cell size with high-quality resampling.
                img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            else:
                img = Image.new("RGB", (cell_size, cell_size), color=(0, 0, 0))
            cells.append(img)

        canvas = Image.new("RGB", (cell_size * 2, cell_size * 2), color=(0, 0, 0))
        canvas.paste(cells[0], (0, 0))                            # top-left
        canvas.paste(cells[1], (cell_size, 0))                    # top-right
        canvas.paste(cells[2], (0, cell_size))                    # bottom-left
        canvas.paste(cells[3], (cell_size, cell_size))            # bottom-right
        return canvas

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
        component_paths: List[Optional[str]],
        final_snapshot_path: Optional[str] = None,
        debug_collage_save_path: Optional[str] = None,
    ) -> str:
        """Run one MLLM forward pass for a single modeling step.

        Parameters
        ----------
        component_paths:
            Exactly 4 paths in canonical row order
            (prev_depth_map, sketch_plane_mask, reference_mask, result_frame).
            ``None`` is allowed for any missing component.
        final_snapshot_path:
            Optional path to the part's ``final_snapshot.png``. If provided
            it is shown AFTER the collage as supplementary context with an
            explicit text label so the model does not confuse it with the
            primary input. **Off by default** -- the system prompt is
            written for a single 2 x 2 collage.
        debug_collage_save_path:
            If set, save the assembled collage to this path (useful for
            eyeballing what the model actually sees).

        Returns
        -------
        str
            A cleaned-up single-sentence description in the format
            ``"<Operation>: Based on <sketch/ref>, generated <changes>."``.
        """
        torch = self._torch

        # ---- 1) Build the 2 x 2 collage and stash for reuse ----
        collage = self._build_collage_image(
            component_paths, cell_size=self.cfg.collage_cell_size
        )
        if debug_collage_save_path is not None:
            parent = os.path.dirname(debug_collage_save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            collage.save(debug_collage_save_path)

        # ---- 2) Compose the chat messages ----
        # qwen-vl-utils handles PIL images in the "image" field directly.
        user_content: List[dict] = [
            {"type": "image", "image": collage},
        ]
        if final_snapshot_path is not None and os.path.isfile(final_snapshot_path):
            user_content.append({
                "type": "text",
                "text": (
                    "Supplementary global context: the part's final shape (NOT "
                    "part of the 2x2 collage above)."
                ),
            })
            user_content.append({"type": "image", "image": self._file_uri(final_snapshot_path)})
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


def collect_step_images(step_dir: str) -> List[Optional[str]]:
    """Return 4 absolute paths in canonical row order, with ``None`` for missing files.

    Order matches :data:`config.ROW_FILENAMES`, which in turn maps 1-to-1 to
    the 2 x 2 collage cell layout:

        [0] -> top-left      (prev_depth_map.png)
        [1] -> top-right     (sketch_plane_mask.png)
        [2] -> bottom-left   (reference_mask.png)
        [3] -> bottom-right  (result_frame.png -- LOCAL feature wireframe)
    """
    out: List[Optional[str]] = []
    for filename in ROW_FILENAMES:
        path = os.path.join(step_dir, filename)
        out.append(os.path.abspath(path) if os.path.isfile(path) else None)
    return out


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
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"data_root not found: {cfg.data_root}")

    part_ids = discover_part_ids(cfg.data_root, cfg.view_suffix)
    if cfg.max_parts is not None:
        part_ids = part_ids[: cfg.max_parts]
    logger.info(
        "Found %d parts with view '%s' under '%s'.",
        len(part_ids), cfg.view_suffix, cfg.data_root,
    )
    if not part_ids:
        logger.warning("Nothing to do. Exiting.")
        return

    # Build a plan first so --dry-run never instantiates the model.
    plan: List[Tuple[str, int, str]] = []  # (part_id, idx, step_dir)
    for part_id in part_ids:
        view_dir = os.path.join(cfg.data_root, f"{part_id}_{cfg.view_suffix}")
        for idx in discover_step_indices(view_dir):
            step_dir = os.path.join(view_dir, f"{STEP_DIR_PREFIX}{idx}")
            plan.append((part_id, idx, step_dir))

    logger.info("Planned %d (part, step) pairs.", len(plan))
    if cfg.dry_run:
        for part_id, idx, step_dir in plan[:20]:
            logger.info("DRY-RUN would label: %s step=%d", step_dir, idx)
        if len(plan) > 20:
            logger.info("... and %d more.", len(plan) - 20)
        return

    labeler = Qwen25VLLabeler(cfg)

    total = 0
    written = 0
    skipped = 0
    broadcast = 0
    failures = 0
    t0 = time.time()

    pbar = tqdm(plan, desc="auto-labeling", smoothing=0.05)
    for part_id, idx, step_dir in pbar:
        total += 1
        prompt_path = os.path.join(step_dir, PROMPT_FILENAME)

        # Resume: skip steps already labeled (unless --overwrite).
        if (
            os.path.isfile(prompt_path)
            and os.path.getsize(prompt_path) > 0
            and not cfg.overwrite
        ):
            skipped += 1
            continue

        component_paths = collect_step_images(step_dir)
        if all(p is None for p in component_paths):
            logger.warning("No component images in %s; skipping.", step_dir)
            failures += 1
            continue
        # The bottom-right cell (result_frame.png) is the most informative
        # for the system prompt's color-coded analysis; warn if absent.
        if component_paths[3] is None:
            logger.warning(
                "Missing result_frame.png for %s/step_%d -- collage bottom-right "
                "will be blank; output will likely be unreliable.",
                part_id, idx,
            )

        # Optional global reference image (off by default for the new prompt).
        final_snapshot: Optional[str] = None
        if cfg.include_final_snapshot:
            cand = os.path.join(
                cfg.data_root, f"{part_id}_{cfg.view_suffix}", I_FINAL_FILENAME,
            )
            if os.path.isfile(cand):
                final_snapshot = cand

        debug_save: Optional[str] = None
        if cfg.debug_collage_dir:
            debug_save = os.path.join(
                cfg.debug_collage_dir,
                f"{part_id}_{cfg.view_suffix}__step_{idx:04d}.png",
            )

        try:
            description = labeler.describe_step(
                component_paths,
                final_snapshot_path=final_snapshot,
                debug_collage_save_path=debug_save,
            )
        except Exception as exc:
            logger.error("MLLM call failed for %s/step_%d: %s", part_id, idx, exc)
            failures += 1
            continue

        if not description:
            logger.warning("Empty description for %s/step_%d; skipping.", part_id, idx)
            failures += 1
            continue

        with open(prompt_path, "w", encoding="utf-8") as fp:
            fp.write(description + "\n")
        written += 1

        if cfg.broadcast_to_all_views:
            broadcast += maybe_broadcast(
                cfg.data_root, part_id, cfg.view_suffix,
                idx, description, cfg.overwrite,
            )

        if (written + failures) % cfg.log_every == 0:
            elapsed = time.time() - t0
            ips = (written + failures) / max(1.0, elapsed)
            pbar.set_postfix(
                wrote=written, skipped=skipped, failed=failures,
                broadcast=broadcast, ips=f"{ips:.2f}",
            )

    logger.info(
        "Done. %d total, %d written, %d skipped, %d broadcast, %d failed. (%.1fs)",
        total, written, skipped, broadcast, failures, time.time() - t0,
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
                   help="Which view folder to read images from (prompts are view-invariant).")
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
    p.add_argument("--collage-cell-size", type=int, default=512,
                   help="Per-cell pixel size of the 2x2 collage. Final collage = 2x2 * cell.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if prompt.txt already exists.")
    p.add_argument("--broadcast", action="store_true",
                   help="Copy each generated prompt.txt to the other 7 view folders.")
    p.add_argument("--include-final-snapshot", action="store_true",
                   help="Append the part's final_snapshot.png as supplementary "
                        "context AFTER the 2x2 collage. Off by default.")
    p.add_argument("--no-flash-attn", action="store_true",
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
        collage_cell_size=args.collage_cell_size,
        overwrite=args.overwrite,
        broadcast_to_all_views=args.broadcast,
        include_final_snapshot=args.include_final_snapshot,
        use_flash_attention=not args.no_flash_attn,
        max_parts=args.max_parts,
        debug_collage_dir=args.debug_collage_dir,
        dry_run=args.dry_run,
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
