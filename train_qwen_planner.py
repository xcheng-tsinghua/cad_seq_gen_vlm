"""Phase 3 — Qwen2.5-VL planner fine-tuning (stub).

Trains (LoRA / full SFT) so that at inference the model predicts the next
painter ``prompt`` from ``final_snapshot`` + ``prev_state`` **without**
``operation_param.json``.

A full implementation would wire:

* :class:`qwen_planner_dataset.QwenPlannerSFTDataset` → chat template + vision batching
* ``transformers`` Trainer or ``trl`` SFT / GRPO on ``Qwen2_5_VLForConditionalGeneration``
* Optional PEFT LoRA on attention + MLP

This module only validates the dataset layout and documents the hook points.

// MVP workflow stub — replace ``NotImplemented`` with your trainer.
"""

from __future__ import annotations

import argparse
import logging
import sys

import config as _cfg  # noqa: F401  — HF cache env

from qwen_planner_dataset import QwenPlannerSFTDataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 Qwen planner SFT (dataset smoke).")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--part-ids-file", type=str, default=None)
    p.add_argument(
        "--prev-state-mode",
        type=str,
        default="depth_before_next",
        choices=("depth_before_next", "overlay_after_prev"),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    try:
        ds = QwenPlannerSFTDataset(
            data_root=args.data_root,
            prev_state_mode=args.prev_state_mode,  # type: ignore[arg-type]
            part_ids_file=args.part_ids_file,
        )
    except RuntimeError as e:
        logging.error("%s", e)
        sys.exit(1)
    logging.info("Dataset OK: %d supervised transitions.", len(ds))
    if len(ds):
        row = ds[0]
        logging.info("Example keys: %s", sorted(row.keys()))


if __name__ == "__main__":
    main()
