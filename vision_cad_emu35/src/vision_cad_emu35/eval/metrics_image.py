from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def l1(pred: Image.Image, gt: Image.Image) -> float:
    return float(np.mean(np.abs(image_to_array(pred) - image_to_array(gt))))


def mse(pred: Image.Image, gt: Image.Image) -> float:
    diff = image_to_array(pred) - image_to_array(gt)
    return float(np.mean(diff * diff))


def psnr(pred: Image.Image, gt: Image.Image) -> float:
    value = mse(pred, gt)
    if value == 0:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(value)))


def ssim(pred: Image.Image, gt: Image.Image) -> float:
    """Compute SSIM with skimage if installed, otherwise a simple global approximation."""
    pred_arr = image_to_array(pred)
    gt_arr = image_to_array(gt)
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(gt_arr, pred_arr, channel_axis=2, data_range=1.0))
    except Exception:
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = pred_arr.mean()
        mu_y = gt_arr.mean()
        sigma_x = pred_arr.var()
        sigma_y = gt_arr.var()
        sigma_xy = ((pred_arr - mu_x) * (gt_arr - mu_y)).mean()
        return float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)))


def optional_lpips(pred: Image.Image, gt: Image.Image) -> float | None:
    try:
        import torch
        import lpips
    except Exception:
        return None

    loss_fn = lpips.LPIPS(net="alex")
    pred_tensor = torch.from_numpy(image_to_array(pred)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    gt_tensor = torch.from_numpy(image_to_array(gt)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(loss_fn(pred_tensor, gt_tensor).item())


def compute_image_metrics(pred: Image.Image, gt: Image.Image, include_optional: bool = False) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "l1": l1(pred, gt),
        "mse": mse(pred, gt),
        "psnr": psnr(pred, gt),
        "ssim": ssim(pred, gt),
    }
    if include_optional:
        metrics["lpips"] = optional_lpips(pred, gt)
    return metrics

