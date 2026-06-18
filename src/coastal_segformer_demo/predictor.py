from __future__ import annotations

import numpy as np


CLASS_NAMES = ["water", "vegetation", "sand_or_built"]
CLASS_COLORS = np.array(
    [
        [31, 144, 214],
        [51, 140, 84],
        [214, 185, 112],
    ],
    dtype="uint8",
)


def heuristic_logits(bands: np.ndarray) -> np.ndarray:
    """Return deterministic class logits for a public-safe synthetic fixture.

    This is not a trained model. It emulates the shape and post-processing
    contract of segmentation inference so the repository can safely demonstrate
    tiling, stitching and visualization before a public checkpoint is selected.
    """
    blue, green, red, nir = bands
    ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    brightness = (blue + green + red) / 3.0
    water = 2.0 * ndwi - 0.2 * ndvi
    vegetation = 2.0 * ndvi - 0.2 * brightness
    sand = 1.5 * brightness - 0.4 * ndwi - 0.4 * ndvi
    return np.stack([water, vegetation, sand]).astype("float32")


def logits_to_mask(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=0).astype("uint8")

