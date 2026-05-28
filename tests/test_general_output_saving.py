from __future__ import annotations

import json

from PIL import Image

from inference.general import save_general_result


def test_save_general_text_only_result(tmp_path):
    response = save_general_result(
        {
            "raw_text": "hello from emu",
            "images": [],
            "metadata": {"num_generated_images": 0},
            "debug": {"events": [{"text_length": 14}]},
        },
        tmp_path,
    )

    assert response["raw_text"] == "hello from emu"
    assert response["num_generated_images"] == 0
    assert response["image_missing"] is True
    assert (tmp_path / "raw_text.txt").read_text(encoding="utf-8") == "hello from emu"
    assert not (tmp_path / "generated_image.png").exists()
    assert (tmp_path / "emu35_events_debug.json").exists()


def test_save_general_image_only_result(tmp_path):
    image = Image.new("RGB", (16, 16), "red")
    response = save_general_result(
        {
            "raw_text": "",
            "images": [image],
            "metadata": {"num_generated_images": 1},
        },
        tmp_path,
    )

    assert response["raw_text_missing"] is True
    assert response["num_generated_images"] == 1
    assert (tmp_path / "generated_image.png").exists()
    saved = json.loads((tmp_path / "response.json").read_text(encoding="utf-8"))
    assert saved["generated_image_paths"] == [str(tmp_path / "generated_image.png")]


def test_save_general_text_and_image_result(tmp_path):
    image = Image.new("RGB", (16, 16), "blue")
    response = save_general_result(
        {
            "raw_text": "a generated preview is attached",
            "image": image,
            "metadata": {"num_generated_images": 1},
        },
        tmp_path,
        prompt="Describe and edit",
        input_image_count=2,
        latency_seconds=1.25,
    )

    assert response["raw_text_missing"] is False
    assert response["input_image_count"] == 2
    assert response["latency_seconds"] == 1.25
    assert (tmp_path / "prompt.txt").read_text(encoding="utf-8") == "Describe and edit"
    assert (tmp_path / "generated_image.png").exists()
