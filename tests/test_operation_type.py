from __future__ import annotations

import json

from vision_cad_emu35.data.operation_type import get_exact_operation_type_from_param


def _write_param(tmp_path, payload):
    path = tmp_path / "operation_param.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_normal_modeling_type(tmp_path):
    path = _write_param(tmp_path, {"modeling_type": "extrude"})
    assert get_exact_operation_type_from_param(path) == "extrude"


def test_other_raw_type(tmp_path):
    path = _write_param(tmp_path, {"modeling_type": "other", "raw_type": "custom_feature"})
    assert get_exact_operation_type_from_param(path) == "custom_feature"


def test_symmetric_operation(tmp_path):
    path = _write_param(tmp_path, {"modeling_type": "extrude", "is_symmetric": True})
    assert get_exact_operation_type_from_param(path) == "symmetric_extrude"


def test_construct_type_new_and_add_map_to_add(tmp_path):
    path = _write_param(tmp_path, {"modeling_type": "extrude", "construct_type": "NEW"})
    assert get_exact_operation_type_from_param(path) == "extrude_add"
    path = _write_param(tmp_path, {"modeling_type": "revolve", "construct_type": "ADD"})
    assert get_exact_operation_type_from_param(path) == "revolve_add"


def test_construct_type_anything_else_maps_to_cut(tmp_path):
    for construct_type in ["REMOVE", "CUT", "SUBTRACT", "unexpected"]:
        path = _write_param(tmp_path, {"modeling_type": "extrude", "construct_type": construct_type})
        assert get_exact_operation_type_from_param(path) == "extrude_cut"


def test_draft_angle_prefix(tmp_path):
    path = _write_param(tmp_path, {"modeling_type": "extrude", "construct_type": "REMOVE", "draft_angle": 2.5})
    assert get_exact_operation_type_from_param(path) == "draft_extrude_cut"

