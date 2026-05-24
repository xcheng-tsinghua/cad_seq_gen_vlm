from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    gpu: dict[str, Any]


class GenerateResponse(BaseModel):
    operation_type: str
    image_base64: str
    metadata: dict[str, Any]
    latency_seconds: float


class BatchResponse(BaseModel):
    results: list[dict[str, Any]]


class AutoregressiveResponse(BaseModel):
    steps: list[dict[str, Any]]

