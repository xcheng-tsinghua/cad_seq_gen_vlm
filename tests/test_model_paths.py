from __future__ import annotations

import pytest

from config import AppConfig, find_project_root, resolve_project_path
from model_paths import apply_model_root_override, validate_local_model_paths
from models.emu35_adapter import build_emu35_generation_cfg
from models.emu35_compat import patch_emu3_tokenizer_file
from utils.runtime_env import normalize_thread_env


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


def test_relative_emu_repo_path_resolves_from_project_root():
    project_root = find_project_root()
    assert resolve_project_path("third_party/Emu3.5") == project_root / "third_party" / "Emu3.5"


def test_thread_env_normalizes_invalid_values(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "auto")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    normalized = normalize_thread_env()
    assert normalized == {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}


def test_emu3_tokenizer_source_patch_adds_safe_helper(tmp_path):
    source = tmp_path / "tokenization_emu3.py"
    source.write_text(
        "class Emu3Tokenizer:\n"
        "    def _add_tokens(self, new_tokens):\n"
        "        for surface_form in new_tokens:\n"
        "            if surface_form not in self.special_tokens_set:\n"
        "                pass\n",
        encoding="utf-8",
    )
    result = patch_emu3_tokenizer_file(source)
    text = source.read_text(encoding="utf-8")
    assert result == {"patch_needed": True, "patch_applied": True}
    assert "_get_emu3_special_tokens_set" in text
    assert "surface_form not in self._get_emu3_special_tokens_set()" in text


def test_build_emu35_generation_cfg_contains_required_fields():
    cfg = build_emu35_generation_cfg({"max_new_tokens": 12, "unconditional_type": "no_text"})
    assert cfg.unconditional_type == "no_text"
    assert cfg.sampling_params["max_new_tokens"] == 12
    assert "text_top_k" in cfg.sampling_params
    assert "image_top_k" in cfg.sampling_params
    assert hasattr(cfg, "special_token_ids")
