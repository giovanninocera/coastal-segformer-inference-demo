from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DemoConfig:
    output_dir: Path
    width: int = 192
    height: int = 192
    seed: int = 31
    tile_size: int = 64
    stride: int = 64


def load_config(path: str | Path) -> DemoConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    output_dir = Path(raw.get("output_dir", "outputs/demo"))
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()
    config = DemoConfig(
        output_dir=output_dir,
        width=int(raw.get("width", 192)),
        height=int(raw.get("height", 192)),
        seed=int(raw.get("seed", 31)),
        tile_size=int(raw.get("tile_size", 64)),
        stride=int(raw.get("stride", 64)),
    )
    validate_config(config)
    return config


def validate_config(config: DemoConfig) -> None:
    if config.width <= 0 or config.height <= 0:
        raise ValueError("width and height must be positive")
    if config.tile_size <= 0 or config.stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if config.tile_size > max(config.width, config.height):
        raise ValueError("tile_size must fit inside the demo image")

