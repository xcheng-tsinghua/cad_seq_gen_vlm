from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_cad_emu35.config import load_config
from vision_cad_emu35.data.dataset import CADStepDataset
from vision_cad_emu35.eval.metrics_color_masks import compute_all_color_metrics
from vision_cad_emu35.eval.metrics_image import compute_image_metrics
from vision_cad_emu35.eval.metrics_text import summarize_text_metrics
from vision_cad_emu35.eval.report import save_confusion_matrix_png, save_metrics, save_qualitative_grid
from vision_cad_emu35.inference.single_step import load_adapter_for_inference
from vision_cad_emu35.utils.jsonl import write_jsonl
from vision_cad_emu35.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate operation type and generated overlay images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    output_dir = Path(args.output_dir or Path(config.data.output_dir) / f"eval_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CADStepDataset(Path(config.data.manifest_dir) / f"{args.split}.jsonl", image_size=config.data.image_size)
    adapter = load_adapter_for_inference(config, args.checkpoint)

    rows = []
    pred_ops: list[str] = []
    target_ops: list[str] = []
    qualitative = []
    failed_dir = output_dir / "failed_cases"
    failed_dir.mkdir(parents=True, exist_ok=True)

    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    for idx in range(limit):
        sample = dataset[idx]
        result = adapter.generate(sample["final_snapshot"], sample["prev_depth_map"], sample["prompt"], config.generation)
        pred_op = result["operation_type"]
        target_op = sample["operation_type"]
        pred_ops.append(pred_op)
        target_ops.append(target_op)
        image_metrics = compute_image_metrics(result["image"], sample["target_image"], include_optional=True)
        color_metrics = compute_all_color_metrics(result["image"], sample["target_image"])
        row = {
            "sample_id": sample.get("sample_id"),
            "pred_operation": pred_op,
            "target_operation": target_op,
            "image_metrics": image_metrics,
            "color_metrics": color_metrics,
        }
        rows.append(row)
        if pred_op != target_op:
            case_dir = failed_dir / f"{idx:05d}_{sample.get('sample_id', 'sample')}"
            case_dir.mkdir(parents=True, exist_ok=True)
            result["image"].save(case_dir / "pred_overlayed_all.png")
            sample["target_image"].save(case_dir / "target_overlayed_all.png")
            (case_dir / "result.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        if len(qualitative) < 16:
            qualitative.append({**row, "pred_image": result["image"], "gt_image": sample["target_image"]})

    text_metrics = summarize_text_metrics(pred_ops, target_ops)
    avg_image = _average_nested([row["image_metrics"] for row in rows])
    avg_color = _average_nested([row["color_metrics"] for row in rows])
    metrics = {"text": text_metrics, "image": avg_image, "color_masks": avg_color, "num_samples": len(rows)}
    save_metrics(metrics, output_dir)
    save_confusion_matrix_png(text_metrics["labels"], text_metrics["confusion_matrix"], output_dir / "operation_confusion_matrix.png")
    save_qualitative_grid(qualitative, output_dir / "qualitative_grid.png")
    write_jsonl(output_dir / "per_sample_results.jsonl", rows)
    print(json.dumps(metrics, indent=2, default=str))


def _average_nested(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


if __name__ == "__main__":
    main()

