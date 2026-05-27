from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from data.operation_type import get_exact_operation_type_from_param
from utils.image_io import validate_image_file


REQUIRED_STEP_FILES = (
    "prev_depth_map.png",
    "current_depth_map.png",
    "operation_param.json",
    "overlayed_all.png",
)


@dataclass
class ScanIssue:
    path: str
    issue: str
    severity: str = "warning"


@dataclass
class DatasetScanResult:
    samples: list[dict[str, Any]]
    issues: list[ScanIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def parse_part_view_name(name: str) -> tuple[str, str]:
    """Split [CAD_PART_ID]_[VIEW_SUFFIX] using the last underscore."""
    if "_" not in name:
        return name, ""
    cad_part_id, view_suffix = name.rsplit("_", 1)
    return cad_part_id, view_suffix


def rollback_index_from_name(name: str) -> int | None:
    match = re.fullmatch(r"roll_back_index_(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def scan_dataset(
    dataset_root: str | Path,
    add_stop_samples: bool = True,
    stop_image_policy: str = "copy_last_depth",
    validate_images: bool = True,
) -> DatasetScanResult:
    """Scan a CAD rollback dataset and return manifest-ready samples."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: list[dict[str, Any]] = []
    issues: list[ScanIssue] = []
    part_ids: set[str] = set()
    view_dirs = 0
    rollback_steps = 0
    op_hist: Counter[str] = Counter()

    for part_view_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        final_snapshot = part_view_dir / "final_snapshot.png"
        if not final_snapshot.exists():
            continue

        view_dirs += 1
        cad_part_id, view_suffix = parse_part_view_name(part_view_dir.name)
        part_ids.add(cad_part_id)
        if validate_images and not validate_image_file(final_snapshot):
            issues.append(ScanIssue(str(final_snapshot), "final_snapshot.png is missing or corrupted", "error"))

        rollback_dirs = []
        for child in part_view_dir.iterdir():
            if not child.is_dir():
                continue
            idx = rollback_index_from_name(child.name)
            if idx is not None:
                rollback_dirs.append((idx, child))
        rollback_dirs.sort(key=lambda item: item[0])

        valid_samples_for_view: list[dict[str, Any]] = []
        for rollback_index, rollback_dir in rollback_dirs:
            missing = [name for name in REQUIRED_STEP_FILES if not (rollback_dir / name).exists()]
            if missing:
                issues.append(
                    ScanIssue(
                        str(rollback_dir),
                        f"Missing required files: {', '.join(missing)}",
                        "error",
                    )
                )
                continue

            image_paths = [
                rollback_dir / "prev_depth_map.png",
                rollback_dir / "current_depth_map.png",
                rollback_dir / "overlayed_all.png",
            ]
            if validate_images:
                bad_images = [str(path) for path in image_paths if not validate_image_file(path)]
                if bad_images:
                    issues.append(
                        ScanIssue(str(rollback_dir), f"Corrupted image files: {', '.join(bad_images)}", "error")
                    )
                    continue

            param_path = rollback_dir / "operation_param.json"
            try:
                operation_type = get_exact_operation_type_from_param(param_path)
            except Exception as exc:
                issues.append(ScanIssue(str(param_path), f"Failed to derive operation type: {exc}", "error"))
                continue

            sample = {
                "sample_id": f"{part_view_dir.name}__roll_back_index_{rollback_index}",
                "final_snapshot_path": str(final_snapshot),
                "prev_depth_map_path": str(rollback_dir / "prev_depth_map.png"),
                "overlayed_all_path": str(rollback_dir / "overlayed_all.png"),
                "current_depth_map_path": str(rollback_dir / "current_depth_map.png"),
                "operation_param_path": str(param_path),
                "operation_type": operation_type,
                "cad_part_id": cad_part_id,
                "view_suffix": view_suffix,
                "part_view_id": part_view_dir.name,
                "rollback_index": rollback_index,
                "is_stop_sample": False,
            }
            valid_samples_for_view.append(sample)
            samples.append(sample)
            op_hist[operation_type] += 1
            rollback_steps += 1

        if add_stop_samples and valid_samples_for_view:
            last = valid_samples_for_view[-1]
            stop_overlay = (
                last["current_depth_map_path"]
                if stop_image_policy == "copy_last_depth"
                else ""
            )
            stop_sample = {
                "sample_id": f"{part_view_dir.name}__stop",
                "final_snapshot_path": str(final_snapshot),
                "prev_depth_map_path": last["current_depth_map_path"],
                "overlayed_all_path": stop_overlay,
                "current_depth_map_path": last["current_depth_map_path"],
                "operation_param_path": "",
                "operation_type": "<STOP>",
                "cad_part_id": cad_part_id,
                "view_suffix": view_suffix,
                "part_view_id": part_view_dir.name,
                "rollback_index": None,
                "is_stop_sample": True,
                "stop_image_policy": stop_image_policy,
            }
            samples.append(stop_sample)
            op_hist["<STOP>"] += 1

    stats = {
        "num_parts": len(part_ids),
        "num_views": view_dirs,
        "num_rollback_steps": rollback_steps,
        "num_samples": len(samples),
        "operation_type_histogram": dict(sorted(op_hist.items())),
        "num_issues": len(issues),
        "num_error_issues": sum(1 for issue in issues if issue.severity == "error"),
    }
    return DatasetScanResult(samples=samples, issues=issues, stats=stats)

