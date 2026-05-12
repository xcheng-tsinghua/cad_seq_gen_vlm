from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = torch.sum(probs * targets, dim=(1, 2, 3))
    union = torch.sum(probs + targets - probs * targets, dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()


class SobelGrad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return gx, gy


class StructuredLoss(nn.Module):
    """Loss for structured multi-head output.

    outputs and targets: [B,4,H,W], head order
    0 prev_depth_map, 1 sketch_plane_mask, 2 reference_mask, 3 result_frame.
    """

    def __init__(
        self,
        w_depth: float = 1.0,
        w_sketch_mask: float = 1.0,
        w_ref_mask: float = 1.0,
        w_wire: float = 1.0,
        w_iou: float = 0.7,
        w_wire_edge: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_depth = w_depth
        self.w_sketch_mask = w_sketch_mask
        self.w_ref_mask = w_ref_mask
        self.w_wire = w_wire
        self.w_iou = w_iou
        self.w_wire_edge = w_wire_edge
        self.sobel = SobelGrad()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # prev_depth_map (regression-like)
        loss_depth = F.smooth_l1_loss(torch.sigmoid(logits[:, 0:1]), targets[:, 0:1])

        # mask heads: BCE + soft IoU
        loss_sketch_bce = F.binary_cross_entropy_with_logits(logits[:, 1:2], targets[:, 1:2])
        loss_ref_bce = F.binary_cross_entropy_with_logits(logits[:, 2:3], targets[:, 2:3])
        loss_sketch_iou = soft_iou_loss(logits[:, 1:2], targets[:, 1:2])
        loss_ref_iou = soft_iou_loss(logits[:, 2:3], targets[:, 2:3])
        loss_sketch = loss_sketch_bce + self.w_iou * loss_sketch_iou
        loss_ref = loss_ref_bce + self.w_iou * loss_ref_iou

        # result_frame: BCE + IoU + edge-consistency
        wire_logits = logits[:, 3:4]
        wire_targets = targets[:, 3:4]
        loss_wire_bce = F.binary_cross_entropy_with_logits(wire_logits, wire_targets)
        loss_wire_iou = soft_iou_loss(wire_logits, wire_targets)
        wire_probs = torch.sigmoid(wire_logits)
        pred_gx, pred_gy = self.sobel(wire_probs)
        gt_gx, gt_gy = self.sobel(wire_targets)
        loss_wire_edge = F.l1_loss(pred_gx, gt_gx) + F.l1_loss(pred_gy, gt_gy)
        loss_wire = loss_wire_bce + self.w_iou * loss_wire_iou + self.w_wire_edge * loss_wire_edge

        total = (
            self.w_depth * loss_depth
            + self.w_sketch_mask * loss_sketch
            + self.w_ref_mask * loss_ref
            + self.w_wire * loss_wire
        )
        return {
            "total": total,
            "depth": loss_depth.detach(),
            "sketch": loss_sketch.detach(),
            "reference": loss_ref.detach(),
            "wire": loss_wire.detach(),
            "wire_edge": loss_wire_edge.detach(),
        }


def sd_latent_consistency_loss(
    vae: nn.Module,
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    head_index: int = 3,
) -> torch.Tensor:
    """Use SD VAE latent distance as perceptual structural constraint.

    Args:
        vae: Diffusers AutoencoderKL (frozen).
        pred_logits: [B,4,H,W]
        target: [B,4,H,W]
        head_index: default 3 for result_frame.
    """
    pred = torch.sigmoid(pred_logits[:, head_index : head_index + 1])
    gt = target[:, head_index : head_index + 1]

    pred_rgb = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
    gt_rgb = gt.repeat(1, 3, 1, 1) * 2.0 - 1.0

    pred_lat = vae.encode(pred_rgb).latent_dist.mean
    with torch.no_grad():
        gt_lat = vae.encode(gt_rgb).latent_dist.mean
    return F.l1_loss(pred_lat, gt_lat)

