"""Phase 1 — Pseudo-labeling with Qwen2.5-VL (painter-oriented prompts).

For each ``roll_back_index_*`` under ``<part>_PPP/``, builds **Phase-1**
training text in :data:`config.PROMPT_FILENAME` for the diffusion painter
(:class:`dataset.CADSingleViewDataset`).

**Inputs to Qwen (semantic–geometric decoupling):**

1. ``prev_depth_map.png`` — clean depth / geometry *before* this step.
2. ``final_snapshot.png`` — global target shape (part root).
3. ``operation_param.json`` — factual constraints (optional via flag).
4. ``overlayed_all.png`` — ground-truth painter target for this step (so the
   caption aligns with yellow/cyan masks and wireframe colors).

**Rule:** Describe how a painter should draw the overlay on the depth base,
consistent with the final part. **No** numerical dimensions and **no** internal
CAD IDs in the output.

Color coding inside ``overlayed_all.png`` (wireframe layer):

    Red     -- reference 2D sketch for this step
    Green   -- newly ADDED solid edges
    Magenta -- REMOVED / CUT edges
    Blue    -- termination face / limit surface

// MVP: single view ``config.MVP_VIEW_SUFFIX`` only.

Usage
-----
::

    pip install -U transformers>=4.49.0 qwen-vl-utils>=0.0.10
    python auto_label.py --data-root ./data
    # Optional:
    #   --no-operation-params    # drop JSON (default: JSON ON)
    #   --overwrite              # regenerate existing prompt.txt
    #   --max-parts N            # smoke test
    #   --dry-run
    #   --debug-overlay-dir DIR

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
from typing import List, Optional, Tuple

from tqdm.auto import tqdm

from config import (
    I_FINAL_FILENAME,
    MVP_VIEW_SUFFIX,
    OPERATION_PARAM_FILENAME,
    OVERLAYED_FILENAME,
    PRETRAINED_DIR,
    PREV_DEPTH_FILENAME,
    PROMPT_FILENAME,
    STEP_DIR_PREFIX,
)


# ===========================================================================
# Static prompts
# ===========================================================================

# // Phase 1: painter metaphor + strict output constraints for SDXL captions.
SYSTEM_PROMPT = """You are helping train a neural "painter" for CAD reverse modeling.

Setup: A painter is given (A) a clean depth-style geometry image — the state *before* this modeling step (`prev_depth_map`) — and (B) the final shaded part (`final_snapshot`). They must learn to paint a composite image that matches the ground-truth target (`overlayed_all`): yellow/cyan masks plus red/green/blue/magenta wireframe overlays on top of the depth base.

You may also see **authoritative operation parameters as JSON**. Use it only to disambiguate operation *type* and topology — translate that into plain visual language. **Never** copy numbers from the JSON into your answer.

**Wireframe / mask semantics in the target overlay:**
- Red: reference 2D sketch driving this step.
- Green: edges of newly **added** solid material.
- Magenta: **cut** / removed material edges.
- Blue: termination / limit surface for the operation.
- Yellow / cyan regions: sketch plane and reference-geometry masks.

**Hard rules for YOUR reply (the painter's instruction text):**
1. Enable someone to paint `overlayed_all` on top of image (A) while staying consistent with global shape (B).
2. Do **not** include numerical dimensions (mm, degrees, counts, etc.).
3. Do **not** name internal CAD element IDs, database keys, or opaque feature handles — use spatial relations and visible regions instead.
4. Prefer no more than five tight sentences; no markdown, no bullet lists, no preamble.
5. follow this structure: '[Operation Type]: [Description]"""

USER_INSTRUCTION = (
    "Write the painter-facing instruction text now — only the instruction, nothing else."
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
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    # The new format is more verbose ("Extrusion: Based on ..., generated ...");
    # 96 tokens leaves headroom while still capping cost.
    max_new_tokens: int = 128
    # Qwen processor "dynamic resolution" knobs. The 4-in-1 overlay is a
    # single image whose native size depends on the data-prep stage. We
    # leave plenty of room (max_pixels ~1.1 MP) to avoid downscaling
    # high-resolution overlays.
    min_pixels: int = 256 * 28 * 28        # ~200 k pixels
    max_pixels: int = 1408 * 28 * 28       # ~1.10 M pixels
    overwrite: bool = False
    max_parts: Optional[int] = None
    log_every: int = 25
    use_flash_attention: bool = True
    dry_run: bool = False
    # Optional: copy every overlay we send to Qwen into this folder (under
    # a deterministic name) for offline visual debugging.
    debug_overlay_dir: Optional[str] = None

    # ---- Ground-truth operation parameters ----
    # When True (default), read ``operation_param.json`` from the step folder
    # under ``<part>_<MVP_VIEW_SUFFIX>/`` and feed it to Qwen2.5-VL as text.
    include_operation_params: bool = True
    operation_params_max_chars: int = 4096


# ===========================================================================
# Qwen2.5-VL wrapper
# ===========================================================================
class Qwen25VLLabeler:
    """Qwen2.5-VL — Phase 1 labeling (prev depth + final + JSON + overlay target)."""

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
    def _load_image_rgb(path: Optional[str], placeholder_size: Tuple[int, int] = (256, 256)):
        """Load an image as RGB, or return a gray placeholder."""
        from PIL import Image
        if path is not None and os.path.isfile(path):
            img = Image.open(path)
            if img.mode == "L":
                return img.convert("RGB")
            return img.convert("RGB")
        w, h = placeholder_size
        return Image.new("RGB", (w, h), color=(32, 32, 32))

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
        prev_depth_path: Optional[str],
        overlay_path: Optional[str],
        final_snapshot_path: Optional[str],
        debug_overlay_save_path: Optional[str] = None,
        params_json_text: Optional[str] = None,
    ) -> str:
        """Phase 1: painter caption for ``overlayed_all`` given depth + final + JSON.

        Image order in chat: prev depth → final snapshot → ground-truth overlay target.
        """
        torch = self._torch

        prev_rgb = self._load_image_rgb(prev_depth_path)
        overlay = self._load_overlay_image(overlay_path)
        final_rgb = self._load_image_rgb(final_snapshot_path)

        if debug_overlay_save_path is not None:
            parent = os.path.dirname(debug_overlay_save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            overlay.save(debug_overlay_save_path)

        user_content: List[dict] = [
            {
                "type": "text",
                "text": (
                    "Image (A) — `prev_depth_map.png`: clean geometry *before* this step "
                    "(grayscale depth; painter starts from here)."
                ),
            },
            {"type": "image", "image": prev_rgb},
            {
                "type": "text",
                "text": (
                    "Image (B) — `final_snapshot.png`: global target appearance of the finished part."
                ),
            },
            {"type": "image", "image": final_rgb},
            {
                "type": "text",
                "text": (
                    "Image (C) — `overlayed_all.png`: **ground-truth** painter target for this step "
                    "(masks + wireframe your text must explain)."
                ),
            },
            {"type": "image", "image": overlay},
        ]

        if params_json_text:
            user_content.append({
                "type": "text",
                "text": (
                    "Authoritative operation parameters (facts only — do not quote numbers or IDs in your reply):\n"
                    "```json\n" + params_json_text + "\n```"
                ),
            })

        user_content.append({"type": "text", "text": USER_INSTRUCTION})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

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

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )

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
# Ground-truth operation parameters loader
# ===========================================================================
def load_operation_params(
    data_root: str,
    part_id: str,
    roll_back_index: int,
    view_suffix: str = MVP_VIEW_SUFFIX,
    max_chars: int = 4096,
) -> Optional[str]:
    """Read ``operation_param.json`` from ``<part>_<view_suffix>/step/`` and
    return it pretty-printed as a string ready to embed in a chat message.

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
        f"{part_id}_{view_suffix}",
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
def discover_part_ids(data_root: str) -> List[str]:
    """Return all part IDs that have ``<id>_<MVP_VIEW_SUFFIX>`` under ``data_root``."""
    suffix = f"_{MVP_VIEW_SUFFIX}"
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


def collect_prev_depth(step_dir: str) -> Optional[str]:
    path = os.path.join(step_dir, PREV_DEPTH_FILENAME)
    return os.path.abspath(path) if os.path.isfile(path) else None


def collect_step_overlay(step_dir: str) -> Optional[str]:
    """Return absolute path to ``overlayed_all.png`` in this step folder, or ``None``."""
    path = os.path.join(step_dir, OVERLAYED_FILENAME)
    return os.path.abspath(path) if os.path.isfile(path) else None


# ===========================================================================
# Main loop
# ===========================================================================
def run(cfg: LabelerConfig) -> None:
    logger = logging.getLogger("auto_label")

    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"data_root not found: {cfg.data_root}")

    part_ids = discover_part_ids(cfg.data_root)
    if cfg.max_parts is not None:
        part_ids = part_ids[: cfg.max_parts]
    logger.info(
        "Found %d parts (view %s). Operation params: %s.",
        len(part_ids),
        MVP_VIEW_SUFFIX,
        "ON" if cfg.include_operation_params else "OFF",
    )
    if not part_ids:
        logger.warning("Nothing to do. Exiting.")
        return

    # Build a plan first so --dry-run never instantiates the model.
    plan: List[Tuple[str, int, str]] = []
    for part_id in part_ids:
        view_dir = os.path.join(cfg.data_root, f"{part_id}_{MVP_VIEW_SUFFIX}")
        for idx in discover_step_indices(view_dir):
            step_dir = os.path.join(view_dir, f"{STEP_DIR_PREFIX}{idx}")
            plan.append((part_id, idx, step_dir))

    logger.info("Planned %d (part, step) pairs.", len(plan))
    if cfg.dry_run:
        for part_id, idx, step_dir in plan[:20]:
            params_tag = ""
            if cfg.include_operation_params:
                p = load_operation_params(
                    cfg.data_root, part_id, idx,
                    max_chars=cfg.operation_params_max_chars,
                )
                params_tag = f"  params={'yes' if p else 'MISSING'}"
            logger.info("DRY-RUN would label: %s step=%d%s", step_dir, idx, params_tag)
        if len(plan) > 20:
            logger.info("... and %d more.", len(plan) - 20)
        return

    labeler = Qwen25VLLabeler(cfg)

    total = 0
    written = 0
    skipped = 0
    failures = 0
    missing_params = 0
    t0 = time.time()

    pbar = tqdm(plan, desc="auto-labeling", smoothing=0.05)
    for part_id, idx, step_dir in pbar:
        total += 1
        prompt_path = os.path.join(step_dir, PROMPT_FILENAME)

        if (
            os.path.isfile(prompt_path)
            and os.path.getsize(prompt_path) > 0
            and not cfg.overwrite
        ):
            skipped += 1
            continue

        overlay_path = collect_step_overlay(step_dir)
        if overlay_path is None:
            logger.warning(
                "Missing %s in %s -- skipping (model would only see a black image).",
                OVERLAYED_FILENAME,
                step_dir,
            )
            failures += 1
            continue

        prev_depth_path = collect_prev_depth(step_dir)
        if prev_depth_path is None:
            logger.warning("Missing %s in %s — using placeholder.", PREV_DEPTH_FILENAME, step_dir)

        final_snapshot = os.path.join(
            cfg.data_root,
            f"{part_id}_{MVP_VIEW_SUFFIX}",
            I_FINAL_FILENAME,
        )
        if not os.path.isfile(final_snapshot):
            logger.warning(
                "Missing %s for part %s — using placeholder.",
                I_FINAL_FILENAME,
                part_id,
            )
            final_snapshot_path: Optional[str] = None
        else:
            final_snapshot_path = final_snapshot

        params_text: Optional[str] = None
        if cfg.include_operation_params:
            params_text = load_operation_params(
                cfg.data_root,
                part_id,
                idx,
                max_chars=cfg.operation_params_max_chars,
            )
            if params_text is None:
                missing_params += 1

        debug_save: Optional[str] = None
        if cfg.debug_overlay_dir:
            debug_save = os.path.join(
                cfg.debug_overlay_dir,
                f"{part_id}_{MVP_VIEW_SUFFIX}__step_{idx:04d}.png",
            )

        try:
            description = labeler.describe_step(
                prev_depth_path=prev_depth_path,
                overlay_path=overlay_path,
                final_snapshot_path=final_snapshot_path,
                debug_overlay_save_path=debug_save,
                params_json_text=params_text,
            )
        except Exception as exc:
            logger.error(
                "MLLM call failed for %s/%s/step_%d: %s",
                part_id,
                MVP_VIEW_SUFFIX,
                idx,
                exc,
            )
            failures += 1
            continue

        if not description:
            logger.warning(
                "Empty description for %s/%s/step_%d; skipping.",
                part_id,
                MVP_VIEW_SUFFIX,
                idx,
            )
            failures += 1
            continue

        with open(prompt_path, "w", encoding="utf-8") as fp:
            fp.write(description + "\n")
        written += 1

        if (written + failures) % cfg.log_every == 0:
            elapsed = time.time() - t0
            ips = (written + failures) / max(1.0, elapsed)
            pbar.set_postfix(
                wrote=written,
                skipped=skipped,
                failed=failures,
                no_params=missing_params,
                ips=f"{ips:.2f}",
            )

    logger.info(
        "Done. %d total, %d written, %d skipped, %d missing-params, %d failed. (%.1fs)",
        total,
        written,
        skipped,
        missing_params,
        failures,
        time.time() - t0,
    )


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args() -> LabelerConfig:
    p = argparse.ArgumentParser(
        description="Auto-generate prompt.txt for CAD modeling steps with Qwen2.5-VL.",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="/opt/data/private/data_set/onshape/cad_seq_img",
        help=f"Root folder containing <part>_{MVP_VIEW_SUFFIX}/ trees.",
    )
    p.add_argument("--no-operation-params", action="store_true",
                   help="Disable feeding operation_param.json as authoritative "
                        "ground-truth context. (Default: ON.)")
    p.add_argument("--operation-params-max-chars", type=int, default=5000,
                   help="Truncate the rendered JSON beyond this many characters. "
                        "(default: 4096)")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HuggingFace id or local path of the Qwen2.5-VL checkpoint.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda | cuda:0 | cpu | mps")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=300,
                   help="Max generated tokens (the new format is more verbose).")
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=1408 * 28 * 28,
                   help="Upper bound on overlay pixel count after Qwen smart-resize.")
    p.add_argument("--overwrite", action="store_false",
                   help="Re-generate even if prompt.txt already exists.")
    p.add_argument("--no-flash-attn", action="store_true",
                   help="Disable flash-attention-2 (use vanilla SDPA).")
    p.add_argument("--max-parts", type=int, default=None,
                   help="Stop after this many parts (smoke testing).")
    p.add_argument("--debug-overlay-dir", type=str, default=None,
                   help="If set, save a copy of every overlay sent to Qwen "
                        f"(named '<part>_{MVP_VIEW_SUFFIX}__step_NNNN.png') under this "
                        "folder for visual inspection.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit; do not load the MLLM.")
    args = p.parse_args()

    return LabelerConfig(
        data_root=args.data_root,
        model_name_or_path=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        overwrite=args.overwrite,
        use_flash_attention=not args.no_flash_attn,
        max_parts=args.max_parts,
        debug_overlay_dir=args.debug_overlay_dir,
        dry_run=args.dry_run,
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
