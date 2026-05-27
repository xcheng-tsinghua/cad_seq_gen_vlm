from __future__ import annotations

import numpy as np
from PIL import Image

from config import RagConfig
from rag.prompt_builder import RagPromptBuilder
from rag.retriever import RagRetriever


def test_missing_kb_dir_is_empty(tmp_path):
    retriever = RagRetriever(tmp_path / "missing", RagConfig())
    assert retriever.is_empty()
    assert retriever.status()["kb_loaded"] is True
    assert retriever.status()["kb_empty"] is True
    image = Image.new("RGB", (16, 16), "black")
    assert retriever.retrieve(image, image) == []


def test_default_kb_dir_is_absolute_autodl_path():
    assert RagConfig().kb_dir == "/root/autodl-tmp/data/outputs/rag_kb"


def test_empty_kb_items_jsonl_is_empty(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "kb_items.jsonl").write_text("", encoding="utf-8")
    np.save(kb / "embeddings.npy", np.zeros((0, 10), dtype=np.float32))
    retriever = RagRetriever(kb, RagConfig())
    assert retriever.is_empty()


def test_prompt_builder_zero_shot_prompt():
    image = Image.new("RGB", (16, 16), "white")
    prompt = RagPromptBuilder(RagConfig()).build(image, image, [])
    assert prompt.zero_shot
    assert "No retrieved examples are available" in prompt.prompt_text
    assert len(prompt.images) == 2
