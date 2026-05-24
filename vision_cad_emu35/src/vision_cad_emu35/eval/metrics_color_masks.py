from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_THRESHOLD_CONFIG: dict[str, dict[str, Any]] = {
    "yellow": {"rgb": (255, 255, 0), "distance": 110, "min_channel": 80},
    "cyan": {"rgb": (0, 255, 255), "distance": 110, "min_channel": 80},
    "red": {"rgb": (255, 0, 0), "distance": 90, "min_channel": 80},
    "blue": {"rgb": (0, 0, 255), "distance": 90, "min_channel": 80},
    "green": {"rgb": (0, 255, 0), "distance": 90, "min_channel": 80},
    "magenta": {"rgb": (255, 0, 255), "distance": 110, "min_channel": 80},
}


def _as_uint8_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.int32)


def extract_color_mask(
    image: Image.Image,
    color_name: str,
    threshold_config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Extract a CAD preview color mask using RGB distance and channel guards."""
    config = dict(DEFAULT_THRESHOLD_CONFIG)
    if threshold_config:
        for key, value in threshold_config.items():
            if isinstance(value, dict) and key in config:
                merged = dict(config[key])
                merged.update(value)
                config[key] = merged
            else:
                config[key] = value
    if color_name not in config:
        raise KeyError(f"Unknown color name: {color_name}")

    arr = _as_uint8_array(image)
    item = config[color_name]
    target = np.asarray(item["rgb"], dtype=np.int32)
    distance = np.sqrt(np.sum((arr - target) ** 2, axis=-1))
    mask = distance <= float(item.get("distance", 90))
    min_channel = int(item.get("min_channel", 0))
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    if color_name == "yellow":
        mask &= (r >= min_channel) & (g >= min_channel) & (b <= max(180, min_channel))
    elif color_name == "cyan":
        mask &= (g >= min_channel) & (b >= min_channel) & (r <= max(180, min_channel))
    elif color_name == "red":
        mask &= (r >= min_channel) & (r > g) & (r > b)
    elif color_name == "blue":
        mask &= (b >= min_channel) & (b > r) & (b > g)
    elif color_name == "green":
        mask &= (g >= min_channel) & (g > r) & (g > b)
    elif color_name == "magenta":
        mask &= (r >= min_channel) & (b >= min_channel) & (g <= max(180, min_channel))
    return mask.astype(bool)


def binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


def binary_precision_recall_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_all_color_metrics(
    pred_image: Image.Image,
    gt_image: Image.Image,
    threshold_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for color_name in DEFAULT_THRESHOLD_CONFIG:
        pred_mask = extract_color_mask(pred_image, color_name, threshold_config)
        gt_mask = extract_color_mask(gt_image, color_name, threshold_config)
        metrics[f"{color_name}_iou"] = binary_iou(pred_mask, gt_mask)
        prf = binary_precision_recall_f1(pred_mask, gt_mask)
        metrics[f"{color_name}_precision"] = prf["precision"]
        metrics[f"{color_name}_recall"] = prf["recall"]
        metrics[f"{color_name}_f1"] = prf["f1"]
    return metrics
