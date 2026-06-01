from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from data.scan_dataset import DatasetScanResult
from filenames import (
    DATASET_CURRENT_DEPTH_MAP,
    DATASET_FINAL_SNAPSHOT,
    DATASET_OPERATION_PARAM,
    DATASET_OVERLAYED_ALL,
    DATASET_PREV_DEPTH_MAP,
    MANIFEST_ALL,
    MANIFEST_ISSUES,
    MANIFEST_STATS,
    MANIFEST_TEST,
    MANIFEST_TRAIN,
    MANIFEST_VAL,
)
from utils.image_io import load_and_resize, save_image
from utils.jsonl import write_jsonl


def deterministic_split(
    samples: list[dict],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    split_by_part_id: bool = True,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split samples deterministically, grouping by CAD part id by default."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    rng = random.Random(seed)
    if split_by_part_id:
        groups: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            groups[str(sample.get("cad_part_id", sample.get("sample_id", "")))].append(sample)
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_keys = set(keys[:n_train])
        val_keys = set(keys[n_train : n_train + n_val])
        test_keys = set(keys[n_train + n_val :])
        return {
            "train": [s for k in keys if k in train_keys for s in groups[k]],
            "val": [s for k in keys if k in val_keys for s in groups[k]],
            "test": [s for k in keys if k in test_keys for s in groups[k]],
        }

    rows = sorted(samples, key=lambda row: row.get("sample_id", ""))
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    return {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }


def write_manifest_bundle(
    result: DatasetScanResult,
    manifest_dir: str | Path,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    split_by_part_id: bool = True,
    seed: int = 42,
) -> dict[str, Path]:
    output_dir = Path(manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = deterministic_split(
        result.samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_by_part_id=split_by_part_id,
        seed=seed,
    )

    paths = {
        "all": output_dir / MANIFEST_ALL,
        "train": output_dir / MANIFEST_TRAIN,
        "val": output_dir / MANIFEST_VAL,
        "test": output_dir / MANIFEST_TEST,
        "stats": output_dir / MANIFEST_STATS,
        "issues": output_dir / MANIFEST_ISSUES,
    }
    write_jsonl(paths["all"], result.samples)
    for split_name, rows in splits.items():
        write_jsonl(paths[split_name], rows)
    paths["stats"].write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    write_jsonl(paths["issues"], [issue.__dict__ for issue in result.issues])
    return paths


def materialize_preprocessed_cache(
    samples: Iterable[dict],
    cache_dir: str | Path,
    image_size: int = 512,
) -> list[dict]:
    """Resize/pad sample images into a cache and rewrite manifest paths."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_samples: list[dict] = []
    image_outputs = {
        "final_snapshot_path": DATASET_FINAL_SNAPSHOT,
        "prev_depth_map_path": DATASET_PREV_DEPTH_MAP,
        "overlayed_all_path": DATASET_OVERLAYED_ALL,
        "current_depth_map_path": DATASET_CURRENT_DEPTH_MAP,
    }
    for sample in samples:
        row = dict(sample)
        sample_dir = cache_root / str(row.get("sample_id", "sample")).replace("/", "_")
        sample_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in image_outputs.items():
            source = row.get(key)
            if not source:
                continue
            target = sample_dir / filename
            save_image(load_and_resize(source, image_size), target)
            row[key] = str(target)
        param_path = row.get("operation_param_path")
        if param_path:
            target_param = sample_dir / DATASET_OPERATION_PARAM
            shutil.copy2(param_path, target_param)
            row["operation_param_path"] = str(target_param)
        cached_samples.append(row)
    return cached_samples


def load_manifest(path: str | Path) -> list[dict]:
    from utils.jsonl import read_jsonl

    return list(read_jsonl(path))
