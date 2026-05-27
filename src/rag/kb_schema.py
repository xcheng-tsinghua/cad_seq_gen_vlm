from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KBItem:
    sample_id: str
    cad_part_id: str
    view_suffix: str
    rollback_index: int | None
    final_snapshot_path: str
    prev_depth_map_path: str
    current_depth_map_path: str
    overlayed_all_path: str
    operation_param_path: str
    operation_type: str
    text_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: dict[str, Any]) -> "KBItem":
        op = str(sample.get("operation_type", "other"))
        summary = (
            f"CAD part {sample.get('cad_part_id', '')}, view {sample.get('view_suffix', '')}, "
            f"rollback index {sample.get('rollback_index')}, operation {op}."
        )
        return cls(
            sample_id=str(sample.get("sample_id", "")),
            cad_part_id=str(sample.get("cad_part_id", "")),
            view_suffix=str(sample.get("view_suffix", "")),
            rollback_index=sample.get("rollback_index"),
            final_snapshot_path=str(sample.get("final_snapshot_path", "")),
            prev_depth_map_path=str(sample.get("prev_depth_map_path", "")),
            current_depth_map_path=str(sample.get("current_depth_map_path", "")),
            overlayed_all_path=str(sample.get("overlayed_all_path", "")),
            operation_param_path=str(sample.get("operation_param_path", "")),
            operation_type=op,
            text_summary=str(sample.get("text_summary", summary)),
            metadata={
                "part_view_id": sample.get("part_view_id"),
                "is_stop_sample": sample.get("is_stop_sample", False),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def item_from_dict(row: dict[str, Any]) -> KBItem:
    return KBItem(
        sample_id=str(row.get("sample_id", "")),
        cad_part_id=str(row.get("cad_part_id", "")),
        view_suffix=str(row.get("view_suffix", "")),
        rollback_index=row.get("rollback_index"),
        final_snapshot_path=str(row.get("final_snapshot_path", "")),
        prev_depth_map_path=str(row.get("prev_depth_map_path", "")),
        current_depth_map_path=str(row.get("current_depth_map_path", "")),
        overlayed_all_path=str(row.get("overlayed_all_path", "")),
        operation_param_path=str(row.get("operation_param_path", "")),
        operation_type=str(row.get("operation_type", "other")),
        text_summary=str(row.get("text_summary", "")),
        metadata=dict(row.get("metadata") or {}),
    )

