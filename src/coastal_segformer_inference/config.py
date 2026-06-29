from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


ModelBackend = Literal["heuristic", "segformer"]


@dataclass(frozen=True)
class InferenceConfig:
    output_dir: Path
    width: int = 192
    height: int = 120
    seed: int = 31
    tile_size: int = 64
    stride: int = 64
    model_backend: ModelBackend = "heuristic"
    checkpoint: str | None = None
    local_files_only: bool = False


def load_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    output_dir = Path(raw.get("output_dir", "outputs/inference"))
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()
    config = InferenceConfig(
        output_dir=output_dir,
        width=int(raw.get("width", 192)),
        height=int(raw.get("height", 192)),
        seed=int(raw.get("seed", 31)),
        tile_size=int(raw.get("tile_size", 64)),
        stride=int(raw.get("stride", 64)),
        model_backend=str(raw.get("model_backend", "heuristic")).lower(),
        checkpoint=raw.get("checkpoint"),
        local_files_only=bool(raw.get("local_files_only", False)),
    )
    validate_config(config)
    return config


def validate_config(config: InferenceConfig) -> None:
    if config.width <= 0 or config.height <= 0:
        raise ValueError("width and height must be positive")
    if config.tile_size <= 0 or config.stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if config.tile_size > max(config.width, config.height):
        raise ValueError("tile_size must fit inside the inference image")
    if config.model_backend not in {"heuristic", "segformer"}:
        raise ValueError("model_backend must be 'heuristic' or 'segformer'")
    if config.model_backend == "segformer" and not config.checkpoint:
        raise ValueError("checkpoint is required when model_backend is 'segformer'")
