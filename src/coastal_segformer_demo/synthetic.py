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

    shoreline = height * 0.66 + 10 * np.sin(xx / 18.0) + 4 * np.cos(xx / 7.0)
    water = yy > shoreline
    lagoon = ((xx - width * 0.34) / (width * 0.24)) ** 2 + ((yy - height * 0.48) / (height * 0.17)) ** 2 < 1.0
    channel = np.abs(yy - (height * 0.55 + 0.075 * xx + 4 * np.sin(xx / 18.0))) < 4
    water = water | lagoon | channel
    vegetation = (
        (((xx - width * 0.70) / (width * 0.16)) ** 2 + ((yy - height * 0.34) / (height * 0.20)) ** 2 < 1.0)
        | (((xx - width * 0.22) / (width * 0.18)) ** 2 + ((yy - height * 0.25) / (height * 0.16)) ** 2 < 1.0)
    ) & ~water
    sand = ~(water | vegetation)
    texture = 0.012 * np.sin(xx / 8.0) + 0.010 * np.cos(yy / 11.0)

    blue = np.full((height, width), 0.15, dtype="float32") + texture
    green = np.full((height, width), 0.17, dtype="float32") + texture
    red = np.full((height, width), 0.18, dtype="float32") + 0.7 * texture
    nir = np.full((height, width), 0.27, dtype="float32") + 0.5 * texture

    blue[water] = 0.16 + 0.02 * np.sin(xx[water] / 14.0)
    green[water] = 0.25 + 0.03 * np.cos(yy[water] / 15.0)
    red[water] = 0.10 + 0.015 * np.sin((xx[water] + yy[water]) / 20.0)
    nir[water] = 0.035 + 0.012 * np.cos(xx[water] / 13.0)

    blue[vegetation] = 0.065 + 0.015 * rng.random(np.count_nonzero(vegetation))
    green[vegetation] = 0.21 + 0.035 * rng.random(np.count_nonzero(vegetation))
    red[vegetation] = 0.075 + 0.020 * rng.random(np.count_nonzero(vegetation))
    nir[vegetation] = 0.43 + 0.050 * rng.random(np.count_nonzero(vegetation))

    blue[sand] = 0.20 + 0.025 * rng.random(np.count_nonzero(sand))
    green[sand] = 0.22 + 0.030 * rng.random(np.count_nonzero(sand))
    red[sand] = 0.24 + 0.030 * rng.random(np.count_nonzero(sand))
    nir[sand] = 0.28 + 0.035 * rng.random(np.count_nonzero(sand))

    noise = rng.normal(0, 0.01, size=(4, height, width)).astype("float32")
    bands = np.clip(np.stack([blue, green, red, nir]) + noise, 0, 1).astype("float32")
    rgb = np.clip(np.stack([bands[2], bands[1], bands[0]], axis=-1) * 520, 0, 255).astype("uint8")
    return SyntheticTile(bands=bands, rgb=rgb)
