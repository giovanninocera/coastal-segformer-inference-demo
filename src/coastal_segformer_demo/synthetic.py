from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticTile:
    bands: np.ndarray
    rgb: np.ndarray


def make_synthetic_multispectral(width: int, height: int, seed: int) -> SyntheticTile:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    water = ((xx - width * 0.38) / (width * 0.30)) ** 2 + ((yy - height * 0.55) / (height * 0.20)) ** 2 < 1.0
    channel = np.abs(yy - (height * 0.60 + 0.08 * xx)) < 4
    vegetation = ((xx - width * 0.68) / (width * 0.18)) ** 2 + ((yy - height * 0.38) / (height * 0.25)) ** 2 < 1.0
    sand = ~(water | channel | vegetation)

    blue = np.full((height, width), 0.13, dtype="float32")
    green = np.full((height, width), 0.15, dtype="float32")
    red = np.full((height, width), 0.16, dtype="float32")
    nir = np.full((height, width), 0.24, dtype="float32")

    water_like = water | channel
    blue[water_like] = 0.18
    green[water_like] = 0.26
    red[water_like] = 0.11
    nir[water_like] = 0.04

    blue[vegetation] = 0.07
    green[vegetation] = 0.20
    red[vegetation] = 0.08
    nir[vegetation] = 0.44

    blue[sand] = 0.20
    green[sand] = 0.22
    red[sand] = 0.24
    nir[sand] = 0.28

    noise = rng.normal(0, 0.01, size=(4, height, width)).astype("float32")
    bands = np.clip(np.stack([blue, green, red, nir]) + noise, 0, 1).astype("float32")
    rgb = np.clip(np.stack([bands[2], bands[1], bands[0]], axis=-1) * 255, 0, 255).astype("uint8")
    return SyntheticTile(bands=bands, rgb=rgb)

