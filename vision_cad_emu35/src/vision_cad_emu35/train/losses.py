from __future__ import annotations

from typing import Any


def scalar_loss_value(loss: Any) -> float:
    try:
        return float(loss.detach().cpu().item())
    except AttributeError:
        return float(loss)

