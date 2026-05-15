"""Pseudo-label generator for CAD modeling steps using Qwen2.5-VL.

For every CAD part under ``--data-root``, this script walks the canonical
view folder (default ``<part>_PPP``), reads the 4 component PNGs of each
``roll_back_index_N`` step, feeds them to **Qwen2.5-VL** with a tightly
constrained instruction, and writes the resulting one-line description to
``prompt.txt`` inside that step folder.

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
    #   --no-final-snapshot                    # skip the global reference image
    #   --max-parts 10                         # quick smoke test
    #   --dry-run                              # print plan, don't load the MLLM

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

# Per-row human-readable labels in the SAME ORDER as ``ROW_FILENAMES`` in
# config.py. Adjust the wording here, not in ``ROW_FILENAMES``.
ROW_LABELS: Tuple[str, ...] = (
    "PREVIOUS DEPTH MAP (the part BEFORE this step)",
    "SKETCH PLANE MASK (the active 2D sketch plane for this step)",
    "REFERENCE MASK (existing geometry referenced or used by this step)",
    "RESULT WIREFRAME (the part AFTER this step)",
)

SYSTEM_PROMPT = (
    "You are a senior mechanical CAD engineer. You analyze single steps of a "
    "parametric CAD modeling sequence and describe each step in concise, "
    "imperative natural language. Use vocabulary common to feature-based "
    "modeling: 'sketch', 'extrude', 'cut', 'revolve', 'sweep', 'loft', "
    "'shell', 'fillet', 'chamfer', 'hole', 'pattern', 'mirror', etc.\n"
    "Answer with EXACTLY ONE imperative sentence shorter than 25 words. "
    "No preamble, no markdown, no bullet points, no quotes."
)

INSTRUCTION_TEXT = (
    "The four images you just saw are the components of ONE modeling step:\n"
    "  1. PREVIOUS DEPTH MAP -- the part BEFORE this step.\n"
    "  2. SKETCH PLANE MASK -- which plane the new sketch sits on.\n"
    "  3. REFERENCE MASK -- which existing geometry is referenced or used.\n"
    "  4. RESULT WIREFRAME -- the part AFTER this step.\n\n"
    "Describe the single CAD operation that transforms the BEFORE state into "
    "the AFTER state. Reply with ONE imperative sentence under 25 words.\n"
    "Examples:\n"
    "  Extrude a 50x30 mm rectangle 20 mm along +Z.\n"
    "  Cut a 10 mm diameter hole through the top face at the centre.\n"
    "  Fillet all top edges with a 2 mm radius."
)

_STEP_RE = re.compile(rf"^{re.escape(STEP_DIR_PREFIX)}(\d+)$")
_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*(here(?:'s| is)?|the|this|sure[,!]?|certainly[,!]?)\b[^.]*[.!?:]\s*", re.IGNORECASE),
    re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*description\s*:\s*", re.IGNORECASE),
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
    max_new_tokens: int = 64
    # Qwen processor "dynamic resolution" knobs. Token cost scales linearly
    # with pixels; with ~5 images per call we keep max_pixels modest.
    min_pixels: int = 256 * 28 * 28        # ~200 k pixels per image
    max_pixels: int = 768 * 28 * 28        # ~600 k pixels per image
    overwrite: bool = False
    broadcast_to_all_views: bool = False
    include_final_snapshot: bool = True
    max_parts: Optional[int] = None
    log_every: int = 25
    use_flash_attention: bool = True
    dry_run: bool = False


# ===========================================================================
# Qwen2.5-VL wrapper
# ===========================================================================
class Qwen25VLLabeler:
    """Thin wrapper around the HF Qwen2.5-VL model.

    Loads model + processor once, then exposes :meth:`describe_step` which
    returns a single cleaned-up string for one (4-image) modeling step.
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
    def _file_uri(abs_path: str) -> str:
        """Build a ``file://`` URI that qwen-vl-utils understands on any OS."""
        path = os.path.abspath(abs_path).replace("\\", "/")
        if not path.startswith("/"):
            path = "/" + path                # Windows: file:///C:/...
        return f"file://{path}"

    @staticmethod
    def _postprocess(text: str) -> str:
        """Trim whitespace, drop preambles, kill stray quoting and bullets."""
        s = text.strip()
        # Strip wrapping quotes if the model added any.
        for q in ("\"", "'", "“", "”", "‘", "’"):
            if s.startswith(q) and s.endswith(q):
                s = s[1:-1].strip()
        # Drop leading bullets / numbering.
        s = re.sub(r"^[-*\u2022\d.)\s]+", "", s)
        # Strip well-known preamble phrases.
        for pattern in _PREAMBLE_PATTERNS:
            s = pattern.sub("", s)
        # Collapse internal newlines into spaces.
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ------------------------------------------------------------------ inference
    def describe_step(
        self,
        labeled_image_paths: List[Tuple[str, str]],
        final_snapshot_path: Optional[str] = None,
    ) -> str:
        """Run one MLLM forward pass for a single modeling step.

        Parameters
        ----------
        labeled_image_paths:
            List of ``(label, abs_path)`` pairs in display order. ``label``
            is shown as a text block immediately *before* the image so the
            MLLM knows what role each picture plays.
        final_snapshot_path:
            Optional path to the part's ``final_snapshot.png``; if provided
            it is shown FIRST as a global "target shape" context image.

        Returns
        -------
        str
            A single-line, cleaned-up description of the CAD operation.
        """
        torch = self._torch

        # ---- Build the multimodal user content ----
        content: List[dict] = []

        if final_snapshot_path is not None and os.path.isfile(final_snapshot_path):
            content.append({
                "type": "text",
                "text": "GLOBAL FINAL SHAPE of the part (target appearance):",
            })
            content.append({"type": "image", "image": self._file_uri(final_snapshot_path)})

        for label, path in labeled_image_paths:
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image", "image": self._file_uri(path)})

        content.append({"type": "text", "text": INSTRUCTION_TEXT})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        # ---- Tokenize + pack vision inputs ----
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

        # ---- Greedy decode ----
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )

        # Slice off the prompt tokens, decode, post-process.
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


def collect_step_images(step_dir: str) -> List[Tuple[str, str]]:
    """Return ``[(label, abs_path), ...]`` for every present component image."""
    out: List[Tuple[str, str]] = []
    for filename, label in zip(ROW_FILENAMES, ROW_LABELS):
        path = os.path.join(step_dir, filename)
        if os.path.isfile(path):
            out.append((label, os.path.abspath(path)))
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

        image_pairs = collect_step_images(step_dir)
        if len(image_pairs) < 1:
            logger.warning("No component images in %s; skipping.", step_dir)
            failures += 1
            continue

        # Optional global reference image.
        final_snapshot: Optional[str] = None
        if cfg.include_final_snapshot:
            cand = os.path.join(
                cfg.data_root, f"{part_id}_{cfg.view_suffix}", I_FINAL_FILENAME,
            )
            if os.path.isfile(cand):
                final_snapshot = cand

        try:
            description = labeler.describe_step(image_pairs, final_snapshot)
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
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if prompt.txt already exists.")
    p.add_argument("--broadcast", action="store_true",
                   help="Copy each generated prompt.txt to the other 7 view folders.")
    p.add_argument("--no-final-snapshot", action="store_true",
                   help="Do not include the part's final_snapshot.png as global context.")
    p.add_argument("--no-flash-attn", action="store_true",
                   help="Disable flash-attention-2 (use vanilla SDPA).")
    p.add_argument("--max-parts", type=int, default=None,
                   help="Stop after this many parts (smoke testing).")
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
        include_final_snapshot=not args.no_final_snapshot,
        use_flash_attention=not args.no_flash_attn,
        max_parts=args.max_parts,
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
