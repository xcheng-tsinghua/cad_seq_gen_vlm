from __future__ import annotations

import os


DEFAULT_THREAD_COUNT = 8


def normalize_thread_env(default_threads: int = DEFAULT_THREAD_COUNT, verbose: bool = False) -> dict[str, str]:
    """Normalize OpenMP/MKL thread env vars before torch or model code is loaded."""

    default_value = str(default_threads if default_threads > 0 else DEFAULT_THREAD_COUNT)
    original_omp = os.environ.get("OMP_NUM_THREADS")
    if not _is_positive_int(original_omp):
        os.environ["OMP_NUM_THREADS"] = default_value

    original_mkl = os.environ.get("MKL_NUM_THREADS")
    if not _is_positive_int(original_mkl):
        os.environ["MKL_NUM_THREADS"] = default_value
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    normalized = {
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "PYTORCH_CUDA_ALLOC_CONF": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
    }
    if verbose:
        print(f"OMP_NUM_THREADS: {original_omp!r} -> {normalized['OMP_NUM_THREADS']!r}")
        print(f"MKL_NUM_THREADS: {original_mkl!r} -> {normalized['MKL_NUM_THREADS']!r}")
    return normalized


def is_positive_int_env(value: str | None) -> bool:
    return _is_positive_int(value)


def _is_positive_int(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False
