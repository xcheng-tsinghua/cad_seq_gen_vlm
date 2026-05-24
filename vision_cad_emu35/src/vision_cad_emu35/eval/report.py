from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def save_metrics(metrics: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_confusion_matrix_png(labels: list[str], matrix: list[list[int]], path: str | Path) -> None:
    cell = 48
    label_w = max(160, min(320, max((len(label) for label in labels), default=1) * 8))
    size = label_w + cell * max(1, len(labels))
    height = label_w + cell * max(1, len(labels))
    img = Image.new("RGB", (size, height), "white")
    draw = ImageDraw.Draw(img)
    max_count = max([max(row) for row in matrix], default=1) or 1
    for i, target in enumerate(labels):
        y = label_w + i * cell
        draw.text((4, y + 14), target[:28], fill="black")
        draw.text((label_w + i * cell + 4, 4), target[:10], fill="black")
        for j, _pred in enumerate(labels):
            x = label_w + j * cell
            count = matrix[i][j]
            shade = 255 - int(180 * count / max_count)
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(shade, shade, 255), outline="gray")
            draw.text((x + 8, y + 14), str(count), fill="black")
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path)


def save_qualitative_grid(rows: list[dict[str, Any]], path: str | Path, max_items: int = 16) -> None:
    if not rows:
        return
    thumbs = []
    for row in rows[:max_items]:
        pred = row.get("pred_image")
        gt = row.get("gt_image")
        if pred is None or gt is None:
            continue
        thumbs.append((pred.convert("RGB").resize((192, 192)), gt.convert("RGB").resize((192, 192)), row))
    if not thumbs:
        return
    width = 384
    height = len(thumbs) * 224
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    for idx, (pred, gt, row) in enumerate(thumbs):
        y = idx * 224
        grid.paste(pred, (0, y + 24))
        grid.paste(gt, (192, y + 24))
        draw.text((4, y + 4), f"P: {row.get('pred_operation')}  T: {row.get('target_operation')}", fill="black")
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(target_path)

