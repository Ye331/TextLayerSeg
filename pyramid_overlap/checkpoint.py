from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_model_checkpoint(model: nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    """Load a full segmentation checkpoint with common wrapper prefixes."""

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if key in payload:
                state = payload[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint must contain a state dictionary")
    state = {
        str(key).removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return {
        "checkpoint": str(path.resolve()),
        "tensors": len(state),
        "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
    }


def save_model_checkpoint(
    model: nn.Module,
    destination: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata or {}}, path)
