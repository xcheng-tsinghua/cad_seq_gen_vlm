from __future__ import annotations

import numpy as np
from PIL import Image

from vision_cad_emu35.eval.metrics_color_masks import (
    binary_iou,
    compute_all_color_metrics,
    extract_color_mask,
)


def test_exact_mask_match_gives_iou_one():
    image = Image.new("RGB", (8, 8), "black")
    image.putpixel((2, 2), (255, 0, 0))
    pred = extract_color_mask(image, "red")
    gt = extract_color_mask(image, "red")
    assert binary_iou(pred, gt) == 1.0


def test_no_overlap_gives_iou_zero():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    pred[0, 0] = True
    gt[3, 3] = True
    assert binary_iou(pred, gt) == 0.0


def test_partial_overlap_gives_intermediate_iou():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    pred[0, 0] = True
    pred[0, 1] = True
    gt[0, 1] = True
    gt[0, 2] = True
    assert 0.0 < binary_iou(pred, gt) < 1.0


def test_compute_all_color_metrics_exact_red_match():
    pred = Image.new("RGB", (8, 8), "black")
    gt = Image.new("RGB", (8, 8), "black")
    pred.putpixel((4, 4), (255, 0, 0))
    gt.putpixel((4, 4), (255, 0, 0))
    metrics = compute_all_color_metrics(pred, gt)
    assert metrics["red_iou"] == 1.0
    assert metrics["red_f1"] == 1.0

