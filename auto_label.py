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

// MVP Refactor: single canonical camera only (see :data:`config.MVP_VIEW_SUFFIX`).
No multi-view iteration, depth-based view picking, or prompt broadcasting.

The output location matches :data:`config.PROMPT_FILENAME`, which is where
:class:`dataset.CADSingleViewDataset._load_prompt` looks for prompts.

Usage
-----
::

    pip install -U transformers>=4.49.0 qwen-vl-utils>=0.0.10
    python auto_label.py --data-root ./data
    # Optional flags:
    #   --model Qwen/Qwen2.5-VL-3B-Instruct   # smaller, faster
    #   --no-operation-params                  # disable JSON grounding (default: ON)
    #   --overwrite                            # re-label even if prompt.txt exists
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
from typing import List, Optional, Tuple

from tqdm.auto import tqdm

from config import (
    I_FINAL_FILENAME,
    MVP_VIEW_SUFFIX,
    OPERATION_PARAM_FILENAME,
    OVERLAYED_FILENAME,
    PRETRAINED_DIR,
    PROMPT_FILENAME,
    STEP_DIR_PREFIX,
)


# ===========================================================================
# Static prompts
# ===========================================================================

SYSTEM_PROMPT = """You are an expert in CAD reverse engineering. I will provide up to two images and a JSON block for one incremental step in a CAD modeling sequence:
1. [Global Context] (optional): The final rendered image of the complete CAD part.
2. [Local Context]: A 4-in-1 composite overlay image representing the CURRENT step.
   - Grayscale base: depth map BEFORE this operation.
   - Semi-transparent YELLOW mask: sketch plane used.
   - Semi-transparent CYAN mask: reference geometry (e.g., sweep path, chanfer edges) (if any).
   - Crisp colored wireframe (Red/Green/Blue/Magenta): the LOCAL feature created/modified in this exact step ONLY.

CRITICAL WIREFRAME COLOR CODING:
- Red: The reference 2D sketch used by this operation.
- Green: Edges of the newly ADDED solid entity.
- Magenta: Edges of the REMOVED / CUT entity.
- Blue: The termination face of this operation.

GROUND-TRUTH OPERATION PARAMETERS (JSON):
Use the provided JSON to determine the exact operation type (e.g., Extrude, Revolve, Cut) and geometric intent. 
RESTRICTIONS ON JSON USAGE: 
1. DO NOT output any specific numerical dimensions (like depth=1.0). 
2. DO NOT output internal CAD software IDs (like "JIB", "FpLunXKtUXBUjWp_0"). If the operation relies on a reference geometry (like an axis of revolution, a sweep path, or a mirror plane), you MUST describe its visual spatial location or orientation based on the images instead of using its raw ID.

YOUR TASK:
Analyze the inputs and write a single, concise sentence describing the operation. 
You MUST establish a Global-Local connection: describe what the local wireframe does, AND explicitly state which specific feature/part of the FINAL CAD model it corresponds to.

Format MUST strictly follow this structure: 
'[Operation Type]: Based on [sketch shape] on the [sketch plane/reference], generated [entity changes], which corresponds to [the specific functional feature/location in the final rendered image].'

For example:
- 'Extrude: Based on the red circular sketch on the yellow-masked sketch plane, extruded a green solid cylinder up to the blue termination face, which corresponds to the main central mounting boss in the final part.'
- 'Extruded Cut: Based on the red rectangular sketch on the yellow-masked sketch plane, cut a magenta negative space up to the blue termination face, which forms the sliding slot on the right side of the final model.'
- 'Fillet: Generated a Magenta solid feature along the edges of the cyan-masked plane, which corresponds to the rounded corners on the base plate of the final part.'"""

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
    # Default OFF: pass ``--include-final-snapshot`` for global reference context.
    include_final_snapshot: bool = False
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
    """Thin wrapper around the HF Qwen2.5-VL model.

    Loads model + processor once, then exposes :meth:`describe_step` which
    feeds a single ``overlayed_all.png`` (the 4-in-1 composite produced by
    data prep) plus optional ground-truth JSON parameters to Qwen2.5-VL,
    and returns a cleaned-up one-line description of the modeling operation.
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
            for offline visual QA of what the model sees.
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

        final_snapshot: Optional[str] = None
        if cfg.include_final_snapshot:
            cand = os.path.join(
                cfg.data_root,
                f"{part_id}_{MVP_VIEW_SUFFIX}",
                I_FINAL_FILENAME,
            )
            if os.path.isfile(cand):
                final_snapshot = cand

        debug_save: Optional[str] = None
        if cfg.debug_overlay_dir:
            debug_save = os.path.join(
                cfg.debug_overlay_dir,
                f"{part_id}_{MVP_VIEW_SUFFIX}__step_{idx:04d}.png",
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
                   help="Upper bound on overlay pixel count after Qwen smart-resize.")
    p.add_argument("--overwrite", action="store_false",
                   help="Re-generate even if prompt.txt already exists.")
    p.add_argument("--include-final-snapshot", action="store_false",
                   help="Also pass final_snapshot.png (global part render) to Qwen.")
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
        include_final_snapshot=args.include_final_snapshot,
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
