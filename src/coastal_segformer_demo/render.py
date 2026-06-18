from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .predictor import CLASS_COLORS, CLASS_NAMES


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_mask(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = render_mask(mask).astype("float32")
    base = rgb.astype("float32")
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype("uint8")


def save_triplet(path: str | Path, rgb: np.ndarray, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Synthetic RGB input", rgb),
        ("Predicted classes", render_mask(mask)),
        ("Overlay", overlay(rgb, mask)),
    ]
    panel_w = rgb.shape[1]
    panel_h = rgb.shape[0]
    canvas = Image.new("RGB", (panel_w * 3, panel_h + 34), (248, 248, 244))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, arr) in enumerate(panels):
        x = idx * panel_w
        canvas.paste(Image.fromarray(arr, mode="RGB"), (x, 34))
        draw.text((x + 8, 10), title, fill=(30, 36, 42), font=_font(14))
    y = panel_h + 12
    for idx, name in enumerate(CLASS_NAMES):
        x = 10 + idx * 140
        draw.rectangle((x, y, x + 14, y + 14), fill=tuple(int(v) for v in CLASS_COLORS[idx]))
        draw.text((x + 20, y), name, fill=(30, 36, 42), font=_font(12))
    canvas.save(path)
    return path

