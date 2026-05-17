"""Smoke test: dataset returns the new 1xV grid for the 4-in-1 layout."""

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
    VIEW_SUFFIXES,
    STEP_DIR_PREFIX,
    ANCHOR_VIEW_SUFFIX,
    OVERLAYED_FILENAME,
    PREV_DEPTH_FILENAME,
    CURRENT_DEPTH_FILENAME,
    OPERATION_PARAM_FILENAME,
    I_FINAL_FILENAME,
    NUM_ROWS,
    NUM_VIEWS,
    TILE_H,
    TILE_W,
    IFINAL_H,
    IFINAL_W,
    PROMPT_FILENAME,
)
from dataset import CADMultiViewDataset, collate_cad_batch, make_worker_init_fn


H, W = 64, 64


def write_image(path, arr, mode="L"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype("uint8"), mode).save(path)


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        parts = ["PART_A", "PART_B"]
        for part in parts:
            for v in VIEW_SUFFIXES:
                base = os.path.join(root, f"{part}_{v}")
                os.makedirs(base, exist_ok=True)
                write_image(
                    os.path.join(base, I_FINAL_FILENAME),
                    np.full((128, 128, 3), 80, dtype=np.uint8),
                    mode="RGB",
                )
                for idx in (1, 3):
                    d = os.path.join(base, f"{STEP_DIR_PREFIX}{idx}")
                    rng = np.random.RandomState(idx * 7 + (1 if v == "NNN" else 0))
                    overlay = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
                    write_image(os.path.join(d, OVERLAYED_FILENAME), overlay, mode="RGB")

                    prev = np.full((H, W), 100, dtype=np.uint8)
                    write_image(os.path.join(d, PREV_DEPTH_FILENAME), prev)
                    curr = prev.copy()
                    if v == ("NNN" if idx == 1 else "NPP"):
                        curr[10:50, 10:50] = 200
                    write_image(os.path.join(d, CURRENT_DEPTH_FILENAME), curr)

            # Prompt + JSON sidecars: anchor view only.
            for idx in (1, 3):
                anchor_step = os.path.join(
                    root,
                    f"{part}_{ANCHOR_VIEW_SUFFIX}",
                    f"{STEP_DIR_PREFIX}{idx}",
                )
                with open(os.path.join(anchor_step, PROMPT_FILENAME), "w", encoding="utf-8") as fp:
                    fp.write(f"Mock prompt for {part} step {idx}.")
                with open(os.path.join(anchor_step, OPERATION_PARAM_FILENAME), "w", encoding="utf-8") as fp:
                    json.dump({"operation_type": "extrusion", "depth_mm": 5.0 + idx}, fp)

        ds = CADMultiViewDataset(data_root=root, random_i_final_view=True, seed=42)
        print(f"len(ds) = {len(ds)}")
        item = ds[0]
        print(f'  I_final  shape = {tuple(item["I_final"].shape)}')
        print(f'  G_prev   shape = {tuple(item["G_prev"].shape)}')
        print(f'  G_target shape = {tuple(item["G_target"].shape)}')
        print(f'  prompt         = {item["prompt"]!r}')
        print(
            f'  sorted_pos/cur/prev = {item["sorted_pos"]}, '
            f'{item["cur_index"]}, {item["prev_index"]}'
        )
        print(f'  i_final_view = {item["i_final_view"]}')

        expected_grid = (3, NUM_ROWS * TILE_H, NUM_VIEWS * TILE_W)
        assert tuple(item["G_target"].shape) == expected_grid, item["G_target"].shape
        assert tuple(item["G_prev"].shape) == expected_grid
        assert tuple(item["I_final"].shape) == (3, IFINAL_H, IFINAL_W)
        for k in ("G_prev", "G_target", "I_final"):
            t = item[k]
            assert torch.is_tensor(t) and t.dtype == torch.float32
            assert t.min() >= -1.001 and t.max() <= 1.001, (k, t.min().item(), t.max().item())

        # First sorted step yields zero G_prev.
        first0 = next(ds[i] for i in range(len(ds)) if ds[i]["sorted_pos"] == 0)
        assert torch.allclose(first0["G_prev"], torch.zeros_like(first0["G_prev"]))

        batch = collate_cad_batch([ds[0], ds[1]])
        assert batch["G_target"].shape == (2,) + expected_grid, batch["G_target"].shape
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
            assert b["G_target"].shape[1:] == expected_grid
            nb += 1
        assert nb == len(ds) // 2
        print(f"DataLoader: iterated {nb} batches OK")
        print("ALL DATASET TESTS PASSED")


if __name__ == "__main__":
    main()
