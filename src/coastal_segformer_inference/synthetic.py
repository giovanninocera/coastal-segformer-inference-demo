from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class SyntheticTile:
    bands: np.ndarray
    rgb: np.ndarray


def _smooth_noise(rng: np.random.Generator, width: int, height: int, scale: int) -> np.ndarray:
    coarse_width = max(6, width // scale)
    coarse_height = max(5, height // scale)
    coarse = rng.normal(0.0, 1.0, size=(coarse_height, coarse_width))
    scaled = ((coarse - coarse.min()) / (np.ptp(coarse) + 1e-6) * 255).astype("uint8")
    field = np.asarray(
        Image.fromarray(scaled).resize((width, height), Image.Resampling.BICUBIC),
        dtype="float32",
    )
    return (field - field.mean()) / (field.std() + 1e-6)


def _lagoon_mask(width: int, height: int) -> np.ndarray:
    geometry_path = Path(__file__).with_name("assets") / "lagoon_shape.png"
    with Image.open(geometry_path) as source:
        geometry = source.convert("L").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(geometry, dtype="uint8") >= 128


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    image = Image.fromarray(mask.astype("uint8") * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(radius * 2 + 1)), dtype="uint8") >= 128


def make_synthetic_multispectral(width: int, height: int, seed: int) -> SyntheticTile:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    broad_texture = _smooth_noise(rng, width, height, 30)
    regional_texture = _smooth_noise(rng, width, height, 15)
    water = _lagoon_mask(width, height)

    marsh_radius = max(4, min(width, height) // 25)
    shore_band = _dilate(water, marsh_radius) & ~water
    marsh_continuity = regional_texture + 0.35 * np.sin(xx / 34.0) - 0.22 * np.cos(yy / 29.0)
    vegetation = shore_band & (marsh_continuity > -0.15)
    bare_land = ~(water | vegetation)

    texture = 0.015 * broad_texture + 0.006 * np.sin(xx / 16.0) + 0.004 * np.cos(yy / 19.0)
    blue = np.full((height, width), 0.19, dtype="float32") + texture
    green = np.full((height, width), 0.21, dtype="float32") + 0.9 * texture
    red = np.full((height, width), 0.24, dtype="float32") + 0.7 * texture
    nir = np.full((height, width), 0.28, dtype="float32") + 0.6 * texture

    water_texture = 0.018 * broad_texture + 0.007 * np.sin((xx + yy) / 18.0)
    water_gradient = 0.012 * (xx / max(width - 1, 1)) - 0.008 * (yy / max(height - 1, 1))
    blue[water] = (0.120 + 0.7 * water_texture + water_gradient)[water]
    green[water] = (0.105 + 0.6 * water_texture + 0.5 * water_gradient)[water]
    red[water] = (0.046 + 0.35 * water_texture)[water]
    nir[water] = (0.024 + 0.20 * water_texture)[water]

    vegetation_texture = 0.014 * regional_texture + 0.004 * np.cos(xx / 12.0)
    blue[vegetation] = (0.060 + 0.45 * vegetation_texture)[vegetation]
    green[vegetation] = (0.205 + vegetation_texture)[vegetation]
    red[vegetation] = (0.072 + 0.45 * vegetation_texture)[vegetation]
    nir[vegetation] = (0.445 + 1.2 * vegetation_texture)[vegetation]

    blue[bare_land] += (0.004 * regional_texture)[bare_land]
    green[bare_land] += (0.003 * regional_texture)[bare_land]
    red[bare_land] += (0.004 * regional_texture)[bare_land]

    bands = np.clip(np.stack([blue, green, red, nir]), 0, 1).astype("float32")
    visible = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    rgb = (np.clip(visible / 0.32, 0, 1) ** 0.88 * 255).astype("uint8")
    return SyntheticTile(bands=bands, rgb=rgb)
