from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .predictor import CLASS_COLORS, CLASS_NAMES


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ["arial.ttf", "segoeui.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _bold_font(size: int) -> ImageFont.ImageFont:
    for candidate in ["arialbd.ttf", "segoeuib.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return _font(size)


def render_mask(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = render_mask(mask).astype("float32")
    base = rgb.astype("float32")
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype("uint8")


def save_triplet(path: str | Path, rgb: np.ndarray, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = 2 if max(rgb.shape[:2]) <= 260 else 1
    panels: list[tuple[str, np.ndarray, int]] = [
        ("Synthetic RGB input", rgb, Image.Resampling.BICUBIC),
        ("Predicted classes", render_mask(mask), Image.Resampling.NEAREST),
        ("Overlay", overlay(rgb, mask, alpha=0.42), Image.Resampling.BICUBIC),
    ]
    panel_w = rgb.shape[1] * scale
    panel_h = rgb.shape[0] * scale
    margin = 24
    gap = 16
    header_h = 76
    label_h = 34
    footer_h = 66
    width = panel_w * 3 + gap * 2 + margin * 2
    height = header_h + label_h + panel_h + footer_h + margin
    canvas = Image.new("RGB", (width, height), (244, 246, 241))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header_h), fill=(21, 44, 61))
    draw.text((margin, 18), "Coastal Segmentation Inference Demo", fill=(255, 255, 255), font=_bold_font(24))
    draw.text(
        (margin, 52),
        "Public-safe synthetic tile, deterministic segmentation contract, stitched full-scene output",
        fill=(207, 224, 229),
        font=_font(13),
    )
    panel_y = header_h + label_h
    for idx, (title, arr, resampling) in enumerate(panels):
        x = margin + idx * (panel_w + gap)
        draw.text((x, header_h + 10), title, fill=(30, 40, 47), font=_bold_font(14))
        image = Image.fromarray(arr, mode="RGB").resize((panel_w, panel_h), resampling)
        canvas.paste(image, (x, panel_y))
        draw.rectangle((x, panel_y, x + panel_w - 1, panel_y + panel_h - 1), outline=(214, 218, 211), width=1)

    y = panel_y + panel_h + 22
    draw.text((margin, y), "Classes", fill=(30, 40, 47), font=_bold_font(13))
    for idx, name in enumerate(CLASS_NAMES):
        x = margin + 74 + idx * 180
        draw.rectangle((x, y, x + 16, y + 16), fill=tuple(int(v) for v in CLASS_COLORS[idx]), outline=(255, 255, 255))
        draw.text((x + 24, y - 1), name, fill=(42, 50, 56), font=_font(12))
    draw.text(
        (margin, y + 30),
        "Not a trained SegFormer result; the figure validates preprocessing, tiling, stitching and visualization boundaries.",
        fill=(83, 91, 96),
        font=_font(11),
    )
    canvas.save(path)
    return path
