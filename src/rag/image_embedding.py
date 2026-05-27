from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.image_io import load_image_rgb


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


class SimpleImageEmbedder:
    """Small CPU-only image embedder for robust default retrieval."""

    def __init__(self, image_size: int = 64, histogram_bins: int = 16) -> None:
        self.image_size = image_size
        self.histogram_bins = histogram_bins
        self.embedding_dim = image_size * image_size + histogram_bins * 3

    def embed_image(self, image: Image.Image | str | Path) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = load_image_rgb(image)
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        gray = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        ).reshape(-1)
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(arr[..., channel], bins=self.histogram_bins, range=(0.0, 1.0), density=False)
            hist = hist.astype(np.float32)
            hist = hist / max(1.0, float(hist.sum()))
            hist_parts.append(hist)
        return l2_normalize(np.concatenate([gray, *hist_parts]).astype(np.float32))

    def embed_pair(self, final_snapshot: Image.Image | str | Path, prev_depth_map: Image.Image | str | Path) -> np.ndarray:
        return l2_normalize(np.concatenate([self.embed_image(final_snapshot), self.embed_image(prev_depth_map)]))


class ClipImageEmbedder:
    """Optional CLIP embedder. Falls back is handled by create_image_embedder."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu") -> None:
        import torch
        import open_clip

        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.model.eval()

    def embed_image(self, image: Image.Image | str | Path) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = load_image_rgb(image)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.detach().cpu().numpy()[0].astype(np.float32)

    def embed_pair(self, final_snapshot: Image.Image | str | Path, prev_depth_map: Image.Image | str | Path) -> np.ndarray:
        return l2_normalize(np.concatenate([self.embed_image(final_snapshot), self.embed_image(prev_depth_map)]))


def create_image_embedder(backend: str = "simple", image_size: int = 64) -> Any:
    if backend == "clip":
        try:
            return ClipImageEmbedder(device="cuda" if _cuda_available() else "cpu")
        except Exception:
            return SimpleImageEmbedder(image_size=image_size)
    return SimpleImageEmbedder(image_size=image_size)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

