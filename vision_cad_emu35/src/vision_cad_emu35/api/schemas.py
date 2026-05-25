from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    kb_loaded: bool
    kb_empty: bool
    kb_item_count: int
    gpu: dict[str, Any]


class GenerateResponse(BaseModel):
    operation_type: str | None
    image_base64: str | None
    retrieved_examples: list[dict[str, Any]]
    zero_shot: bool
    metadata: dict[str, Any]
    latency_seconds: float


class RetrieveResponse(BaseModel):
    retrieved_examples: list[dict[str, Any]]
    zero_shot: bool
    message: str | None = None
