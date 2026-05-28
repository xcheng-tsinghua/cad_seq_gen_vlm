from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    modes_supported: list[str] = Field(default_factory=list)
    kb_dir: str
    kb_loaded: bool
    kb_empty: bool
    kb_item_count: int
    gpu: dict[str, Any]


class GenerateResponse(BaseModel):
    operation_type: str | None
    image_base64: str | None
    retrieved_examples: list[dict[str, Any]]
    kb_dir: str
    zero_shot: bool
    metadata: dict[str, Any]
    latency_seconds: float


class GeneralGenerateResponse(BaseModel):
    raw_text: str
    image_url: str | None
    image_base64: str | None
    generated_image_paths: list[str]
    num_generated_images: int
    metadata: dict[str, Any]
    latency_seconds: float | None = None


class RetrieveResponse(BaseModel):
    kb_dir: str
    retrieved_examples: list[dict[str, Any]]
    zero_shot: bool
    message: str | None = None
