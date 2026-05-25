from __future__ import annotations

import pytest

from vision_cad_emu35.config import AppConfig
from vision_cad_emu35.model_paths import apply_model_root_override, validate_local_model_paths


def test_model_root_override_derives_all_local_paths():
    config = AppConfig()
    apply_model_root_override(config.model, "/new/model/root")
    assert config.model.model_root == "/new/model/root"
    assert config.model.model_id_or_path == "/new/model/root/BAAI/Emu3.5"
    assert config.model.tokenizer_path == "/new/model/root/BAAI/Emu3.5"
    assert config.model.vision_tokenizer_path == "/new/model/root/BAAI/Emu3.5-VisionTokenizer"


def test_local_files_only_missing_paths_fail_early(tmp_path):
    config = AppConfig()
    apply_model_root_override(config.model, tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="python scripts/download_models.py"):
        validate_local_model_paths(config.model)

