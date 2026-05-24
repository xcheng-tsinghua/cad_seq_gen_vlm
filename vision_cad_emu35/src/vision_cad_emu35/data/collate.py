from __future__ import annotations

from typing import Any, Callable


class CADCollator:
    """Collate a batch and optionally hand it to the Emu3.5 adapter."""

    def __init__(self, adapter: Any | None = None) -> None:
        self.adapter = adapter

    def __call__(self, samples: list[dict]) -> dict[str, Any]:
        if self.adapter is None:
            return {"samples": samples}
        if hasattr(self.adapter, "build_training_batch"):
            return self.adapter.build_training_batch(samples)
        return {"samples": [self.adapter.build_training_sample(sample) for sample in samples]}


def identity_collate(samples: list[dict]) -> dict[str, list[dict]]:
    return {"samples": samples}

