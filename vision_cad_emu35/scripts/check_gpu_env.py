from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def main() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print_thread_env_status()
    torch = import_torch()
    if torch is None:
        warn("torch is not installed. Install the Blackwell CUDA 12.8 wheel with requirements-blackwell-cu128.txt.")
        print_optional_imports()
        return

    print("torch import status: ok")
    print(f"torch version: {getattr(torch, '__version__', 'unknown')}")
    cuda_version = getattr(torch.version, "cuda", None)
    print(f"torch.version.cuda: {cuda_version}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available(): {cuda_available}")

    if cuda_version is None:
        warn("torch appears to be CPU-only; torch.version.cuda is None.")
    elif parse_cuda_version(cuda_version) < (12, 8):
        warn(f"torch.version.cuda is lower than 12.8: {cuda_version}")

    if not cuda_available:
        warn("CUDA is unavailable to PyTorch.")
        print_optional_imports()
        return

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    print(f"GPU name: {name}")
    print(f"GPU compute capability: {capability[0]}.{capability[1]}")
    print(f"bf16 supported: {bool(torch.cuda.is_bf16_supported())}")

    allocated = False
    try:
        tensor = torch.ones((8, 8), device="cuda")
        value = float((tensor @ tensor).sum().item())
        del tensor
        torch.cuda.synchronize()
        allocated = True
        print(f"small CUDA tensor allocation: ok ({value:.1f})")
    except Exception as exc:
        warn(f"small CUDA tensor allocation failed: {exc}")

    if capability[0] >= 12 and not allocated:
        warn("GPU compute capability starts with 12 but PyTorch could not allocate a CUDA tensor.")

    print_optional_imports()


def import_torch() -> Any | None:
    try:
        import torch

        return torch
    except Exception as exc:
        print(f"torch import status: failed ({exc})")
        return None


def print_optional_imports() -> None:
    for name in ("flash_attn", "xformers", "bitsandbytes"):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"{name} import status: ok ({version})")
        except Exception as exc:
            warn(f"{name} import status: failed ({exc}); this is optional and standard PyTorch inference will be used.")


def print_thread_env_status() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        print(f"{name}: {value if value is not None else '<unset>'}")
        if not is_positive_integer(value):
            warn(f"{name} is not a positive integer. Runtime scripts will normalize invalid values to 8 before model load.")


def is_positive_integer(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False


def parse_cuda_version(version: str | None) -> tuple[int, int]:
    if not version:
        return (0, 0)
    parts = str(version).split(".")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return (0, 0)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


if __name__ == "__main__":
    main()
