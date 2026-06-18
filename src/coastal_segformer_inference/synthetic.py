from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SyntheticTile:
    bands: np.ndarray
    rgb: np.ndarray


def make_synthetic_multispectral(width: int, height: int, seed: int) -> SyntheticTile:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    noise = rng.normal(0.0, 1.0, size=xx.shape)
    low_freq_noise = np.array(
        Image.fromarray(((noise - noise.min()) / (np.ptp(noise) + 1e-6) * 255).astype("uint8")).resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype="float32",
    )
    low_freq_noise = (low_freq_noise - low_freq_noise.mean()) / (low_freq_noise.std() + 1e-6)

    shoreline = height * 0.66 + 11 * np.sin(xx / 20.0) + 4 * np.cos(xx / 7.5) + 2.5 * low_freq_noise
    water = yy > shoreline
    lagoon = (
        ((xx - width * 0.35) / (width * 0.25)) ** 2
        + ((yy - height * 0.49) / (height * 0.17)) ** 2
        - 0.20 * np.sin(xx / 18.0)
        - 0.08 * np.cos((xx + yy) / 21.0)
        + 0.07 * low_freq_noise
    ) < 1.0
    channel = np.abs(yy - (height * 0.55 + 0.073 * xx + 4.5 * np.sin(xx / 18.0))) < (3.5 + np.sin(xx / 34.0))
    sand_bar = (
        ((xx - width * 0.52) / (width * 0.08)) ** 2
        + ((yy - height * 0.58) / (height * 0.04)) ** 2
        < 1.0
    )
    water = (water | lagoon | channel) & ~sand_bar
    marsh_score = np.sin(xx / 12.0) + np.cos(yy / 15.0) + 0.75 * low_freq_noise
    marsh_band = (~water) & (yy > shoreline - 58) & (yy < shoreline - 8) & (marsh_score > -0.05)
    inland_fields = (
        (~water)
        & (yy < height * 0.48)
        & ((np.sin(xx / 24.0) + np.cos(yy / 19.0) + 0.65 * low_freq_noise) > 0.55)
    )
    riparian_strip = (~water) & (np.abs(yy - (height * 0.55 + 0.073 * xx + 4.5 * np.sin(xx / 18.0))) < 13)
    vegetation = (marsh_band | inland_fields | riparian_strip) & ~water
    sand = ~(water | vegetation)
    built_grid = (
        sand
        & (xx > width * 0.58)
        & (yy < height * 0.42)
        & (((xx.astype(int) % 28) < 3) | ((yy.astype(int) % 24) < 3))
    )
    texture = 0.012 * np.sin(xx / 8.0) + 0.010 * np.cos(yy / 11.0) + 0.010 * low_freq_noise

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
    blue[built_grid] = 0.18 + 0.02 * rng.random(np.count_nonzero(built_grid))
    green[built_grid] = 0.19 + 0.02 * rng.random(np.count_nonzero(built_grid))
    red[built_grid] = 0.22 + 0.02 * rng.random(np.count_nonzero(built_grid))
    nir[built_grid] = 0.24 + 0.02 * rng.random(np.count_nonzero(built_grid))

    noise = rng.normal(0, 0.005, size=(4, height, width)).astype("float32")
    bands = np.clip(np.stack([blue, green, red, nir]) + noise, 0, 1).astype("float32")
    rgb = np.clip(np.stack([bands[2], bands[1], bands[0]], axis=-1) * 520, 0, 255).astype("uint8")
    return SyntheticTile(bands=bands, rgb=rgb)
