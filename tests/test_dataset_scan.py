from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from data.manifest import deterministic_split
from data.scan_dataset import scan_dataset


def _png(path: Path, color=(0, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def _step(root: Path, part_view: str, index: int, op: dict, missing: str | None = None):
    part_dir = root / part_view
    _png(part_dir / "final_snapshot.png", (10, 10, 10))
    step_dir = part_dir / f"roll_back_index_{index}"
    step_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "prev_depth_map.png": lambda: _png(step_dir / "prev_depth_map.png", (20, 20, 20)),
        "current_depth_map.png": lambda: _png(step_dir / "current_depth_map.png", (30, 30, 30)),
        "overlayed_all.png": lambda: _png(step_dir / "overlayed_all.png", (255, 0, 0)),
        "operation_param.json": lambda: (step_dir / "operation_param.json").write_text(json.dumps(op), encoding="utf-8"),
    }
    for name, writer in files.items():
        if name != missing:
            writer()


def test_scan_non_continuous_indices_and_final_snapshot_pairing(tmp_path):
    root = tmp_path / "raw"
    _step(root, "partA_front", 3, {"modeling_type": "revolve", "construct_type": "ADD"})
    _step(root, "partA_front", 1, {"modeling_type": "extrude", "construct_type": "NEW"})

    result = scan_dataset(root, add_stop_samples=False)

    assert [sample["rollback_index"] for sample in result.samples] == [1, 3]
    assert all(Path(sample["final_snapshot_path"]).as_posix().endswith("partA_front/final_snapshot.png") for sample in result.samples)
    assert [sample["operation_type"] for sample in result.samples] == ["extrude_add", "revolve_add"]


def test_missing_files_are_reported(tmp_path):
    root = tmp_path / "raw"
    _step(root, "partB_top", 1, {"modeling_type": "extrude"}, missing="overlayed_all.png")

    result = scan_dataset(root, add_stop_samples=False)

    assert result.samples == []
    assert result.issues
    assert "Missing required files" in result.issues[0].issue


def test_stop_sample_uses_last_current_depth(tmp_path):
    root = tmp_path / "raw"
    _step(root, "partC_iso", 1, {"modeling_type": "extrude", "construct_type": "ADD"})

    result = scan_dataset(root, add_stop_samples=True, stop_image_policy="copy_last_depth")

    stop = result.samples[-1]
    assert stop["operation_type"] == "<STOP>"
    assert stop["prev_depth_map_path"] == result.samples[0]["current_depth_map_path"]
    assert stop["overlayed_all_path"] == result.samples[0]["current_depth_map_path"]


def test_split_by_part_id_has_no_leakage():
    samples = []
    for part_id in ["a", "b", "c", "d", "e"]:
        for view in ["front", "top"]:
            samples.append({"sample_id": f"{part_id}_{view}", "cad_part_id": part_id})

    splits = deterministic_split(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, split_by_part_id=True, seed=7)

    split_parts = {name: {row["cad_part_id"] for row in rows} for name, rows in splits.items()}
    assert split_parts["train"].isdisjoint(split_parts["val"])
    assert split_parts["train"].isdisjoint(split_parts["test"])
    assert split_parts["val"].isdisjoint(split_parts["test"])

