from __future__ import annotations

import os


DEFAULT_THREAD_COUNT = 8


def normalize_thread_env(default_threads: int = DEFAULT_THREAD_COUNT) -> dict[str, str]:
    """Normalize OpenMP/MKL thread env vars before torch or model code is loaded."""

    default_value = str(default_threads if default_threads > 0 else DEFAULT_THREAD_COUNT)
    omp_value = os.environ.get("OMP_NUM_THREADS")
    if not _is_positive_int(omp_value):
        os.environ["OMP_NUM_THREADS"] = default_value
    final_omp = os.environ["OMP_NUM_THREADS"]

    mkl_value = os.environ.get("MKL_NUM_THREADS")
    if not _is_positive_int(mkl_value):
        os.environ["MKL_NUM_THREADS"] = final_omp
    return {
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
    }


def is_positive_int_env(value: str | None) -> bool:
    return _is_positive_int(value)


def _is_positive_int(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False
