from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

INITIAL_THREAD_ENV = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}


def _bootstrap_thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            valid = int(str(os.environ.get(name, "")).strip()) > 0
        except ValueError:
            valid = False
        if not valid:
            os.environ[name] = "8"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_bootstrap_thread_env()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.runtime_env import is_positive_int_env, normalize_thread_env


def main() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print_thread_env_status()
    normalize_thread_env(verbose=True)
    print_package_version("transformers")
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
        original = INITIAL_THREAD_ENV.get(name)
        value = os.environ.get(name)
        print(f"{name}: {value if value is not None else '<unset>'} (original: {original if original is not None else '<unset>'})")
        if not is_positive_int_env(original):
            warn(f"{name} is not a positive integer. Runtime scripts will normalize invalid values to 8 before model load.")


def print_package_version(name: str) -> None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"{name} import status: failed ({exc})")
        return
    print(f"{name} version: {getattr(module, '__version__', 'unknown')}")


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
