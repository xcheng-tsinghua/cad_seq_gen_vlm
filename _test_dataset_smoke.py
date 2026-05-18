"""Smoke test: single-view dataset tensors and collate."""

import os
import sys
import json
import tempfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from config import (
    STEP_DIR_PREFIX,
    MVP_VIEW_SUFFIX,
    OVERLAYED_FILENAME,
    PREV_DEPTH_FILENAME,
    OPERATION_PARAM_FILENAME,
    I_FINAL_FILENAME,
    IFINAL_H,
    IFINAL_W,
    TRAIN_IMAGE_H,
    TRAIN_IMAGE_W,
    PROMPT_FILENAME,
)
from dataset import CADSingleViewDataset, collate_cad_batch, make_worker_init_fn


H, W = 64, 64


def write_image(path, arr, mode="L"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype("uint8"), mode).save(path)


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        parts = ["PART_A", "PART_B"]
        for part in parts:
            base = os.path.join(root, f"{part}_{MVP_VIEW_SUFFIX}")
            os.makedirs(base, exist_ok=True)
            write_image(
                os.path.join(base, I_FINAL_FILENAME),
                np.full((128, 128, 3), 80, dtype=np.uint8),
                mode="RGB",
            )
            for idx in (1, 3):
                d = os.path.join(base, f"{STEP_DIR_PREFIX}{idx}")
                rng = np.random.RandomState(idx * 7)
                overlay = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
                write_image(os.path.join(d, OVERLAYED_FILENAME), overlay, mode="RGB")
                prev = np.full((H, W), 100, dtype=np.uint8)
                write_image(os.path.join(d, PREV_DEPTH_FILENAME), prev)
                with open(os.path.join(d, PROMPT_FILENAME), "w", encoding="utf-8") as fp:
                    fp.write(f"Mock prompt for {part} step {idx}.")
                with open(os.path.join(d, OPERATION_PARAM_FILENAME), "w", encoding="utf-8") as fp:
                    json.dump({"operation_type": "extrusion", "depth_mm": 5.0 + idx}, fp)

        ds = CADSingleViewDataset(data_root=root, train_image_size=(TRAIN_IMAGE_H, TRAIN_IMAGE_W))
        print(f"len(ds) = {len(ds)}")
        item = ds[0]
        print(f'  I_final           shape = {tuple(item["I_final"].shape)}')
        print(f'  condition_image   shape = {tuple(item["condition_image"].shape)}')
        print(f'  target_image      shape = {tuple(item["target_image"].shape)}')
        print(f'  prompt                  = {item["prompt"]!r}')
        print(
            f'  sorted_pos/cur/prev = {item["sorted_pos"]}, '
            f'{item["cur_index"]}, {item["prev_index"]}'
        )

        expected_train = (3, TRAIN_IMAGE_H, TRAIN_IMAGE_W)
        assert tuple(item["target_image"].shape) == expected_train, item["target_image"].shape
        assert tuple(item["condition_image"].shape) == expected_train
        assert tuple(item["I_final"].shape) == (3, IFINAL_H, IFINAL_W)
        for k in ("condition_image", "target_image", "I_final"):
            t = item[k]
            assert torch.is_tensor(t) and t.dtype == torch.float32
            assert t.min() >= -1.001 and t.max() <= 1.001, (k, t.min().item(), t.max().item())

        first0 = next(ds[i] for i in range(len(ds)) if ds[i]["sorted_pos"] == 0)
        assert torch.allclose(first0["condition_image"], torch.zeros_like(first0["condition_image"]))

        batch = collate_cad_batch([ds[0], ds[1]])
        assert batch["target_image"].shape == (2,) + expected_train, batch["target_image"].shape
        assert batch["I_final"].shape == (2, 3, IFINAL_H, IFINAL_W)
        assert isinstance(batch["prompt"], list) and len(batch["prompt"]) == 2

        dl = DataLoader(
            ds,
            batch_size=2,
            num_workers=2,
            worker_init_fn=make_worker_init_fn(base_seed=123),
            collate_fn=collate_cad_batch,
        )
        nb = 0
        for b in dl:
            assert b["target_image"].shape[1:] == expected_train
            nb += 1
        assert nb == len(ds) // 2
        print(f"DataLoader: iterated {nb} batches OK")
        print("ALL DATASET TESTS PASSED")


if __name__ == "__main__":
    main()
