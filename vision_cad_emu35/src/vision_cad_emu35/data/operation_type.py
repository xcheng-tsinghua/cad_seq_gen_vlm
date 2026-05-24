from __future__ import annotations

import json
from pathlib import Path


def get_exact_operation_type_from_param(param_json_path: str | Path) -> str:
    with open(param_json_path, "r") as f:
        param_dict = json.load(f)

    type_str = param_dict["modeling_type"]
    if type_str == "other":
        return param_dict["raw_type"]

    if "is_symmetric" in param_dict and param_dict["is_symmetric"]:
        type_str = "symmetric_" + type_str

    if "construct_type" in param_dict:
        if param_dict["construct_type"] in ("NEW", "ADD"):
            type_str = type_str + "_add"
        else:
            type_str = type_str + "_cut"

    if "draft_angle" in param_dict and param_dict["draft_angle"] is not None:
        type_str = "draft_" + type_str

    return type_str

