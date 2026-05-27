from __future__ import annotations

from typing import Any


def get_gpu_info() -> dict[str, Any]:
    """Return CUDA memory/device information without requiring torch at import time."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "reason": "torch is not installed"}

    if not torch.cuda.is_available():
        return {"cuda_available": False}

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "cuda_available": True,
        "device_index": index,
        "device_name": props.name,
        "total_memory_gb": round(props.total_memory / (1024**3), 3),
        "allocated_gb": round(torch.cuda.memory_allocated(index) / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved(index) / (1024**3), 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(index) / (1024**3), 3),
    }


def empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

