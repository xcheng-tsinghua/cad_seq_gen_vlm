from __future__ import annotations

from pathlib import Path

from PIL import Image

from config import RagConfig
from filenames import DATASET_OVERLAYED_ALL, DATASET_PREV_DEPTH_MAP
from rag.prompt_builder import RagPromptBuilder


def _image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return str(path)


def test_prompt_with_retrieved_examples(tmp_path):
    prev = _image(tmp_path / "e" / DATASET_PREV_DEPTH_MAP, (20, 20, 20))
    overlay = _image(tmp_path / "e" / DATASET_OVERLAYED_ALL, (255, 0, 0))
    example = {
        "sample_id": "s1",
        "score": 0.873,
        "operation_type": "extrude_cut",
        "prev_depth_map_path": prev,
        "overlayed_all_path": overlay,
    }
    query = Image.new("RGB", (16, 16), "white")
    prompt = RagPromptBuilder(RagConfig()).build(query, query, [example])
    assert "Reference Example 1" in prompt.prompt_text
    assert "Operation_Type: extrude_cut" in prompt.prompt_text
    assert len(prompt.images) == 4
    assert not prompt.zero_shot


def test_prompt_without_examples():
    query = Image.new("RGB", (16, 16), "white")
    prompt = RagPromptBuilder(RagConfig()).build(query, query, [])
    assert "zero-shot mode" in prompt.prompt_text
    assert prompt.zero_shot


def test_max_reference_images_respected(tmp_path):
    examples = []
    for idx in range(3):
        prev = _image(tmp_path / str(idx) / "prev.png", (idx, idx, idx))
        overlay = _image(tmp_path / str(idx) / "overlay.png", (255, idx, idx))
        examples.append(
            {
                "sample_id": f"s{idx}",
                "score": 1.0 - idx * 0.1,
                "operation_type": "extrude_add",
                "prev_depth_map_path": prev,
                "overlayed_all_path": overlay,
            }
        )
    cfg = RagConfig(max_reference_images=3)
    prompt = RagPromptBuilder(cfg).build(Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16)), examples)
    assert len(prompt.images) == 5
    assert len(prompt.image_roles) == 5
