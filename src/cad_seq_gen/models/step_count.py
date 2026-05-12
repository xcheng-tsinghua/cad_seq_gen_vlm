from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor
from transformers import CLIPModel, CLIPProcessor


class StepCountPredictor:
    """Predict sequence length from target CAD image via CLIP + KNN."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model.eval()
        self.knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
        self.fitted = False

    @torch.no_grad()
    def _encode(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats[0].detach().cpu().numpy()

    def fit(self, processed_root: Path) -> None:
        processed_root = Path(processed_root)
        step_stats: Dict[str, int] = json.loads((processed_root / "step_stats.json").read_text("utf-8"))
        manifest_path = processed_root / "manifest.jsonl"

        part_to_final_target: Dict[str, Path] = {}
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                part_id = row["part_id"]
                if row["step_index"] == step_stats.get(part_id, -1):
                    part_to_final_target[part_id] = Path(row["target_image"])

        x_list: List[np.ndarray] = []
        y_list: List[float] = []
        for part_id, step_count in step_stats.items():
            final_target = part_to_final_target.get(part_id)
            if final_target is None or not final_target.exists():
                continue
            final_canvas = Image.open(final_target).convert("RGB")
            h, w = final_canvas.size[1], final_canvas.size[0]
            panel = final_canvas.crop((w // 2, h // 2, w, h))  # result_frame panel
            feat = self._encode(panel)
            x_list.append(feat)
            y_list.append(float(step_count))

        if not x_list:
            raise RuntimeError("No valid training samples for step count predictor.")
        self.knn.fit(np.stack(x_list), np.array(y_list))
        self.fitted = True

    def predict(self, image: Image.Image, min_steps: int = 1, max_steps: int = 30) -> int:
        if not self.fitted:
            raise RuntimeError("StepCountPredictor must be fitted before prediction.")
        feat = self._encode(image)
        pred = float(self.knn.predict(feat[None, ...])[0])
        pred = max(float(min_steps), min(float(max_steps), pred))
        return int(round(pred))

