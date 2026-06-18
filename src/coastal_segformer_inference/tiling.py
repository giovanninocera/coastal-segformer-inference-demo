from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    row: int
    col: int
    height: int
    width: int


def make_windows(height: int, width: int, tile_size: int, stride: int) -> list[Window]:
    rows = list(range(0, max(height - tile_size + 1, 1), stride))
    cols = list(range(0, max(width - tile_size + 1, 1), stride))
    if rows[-1] != height - tile_size:
        rows.append(max(height - tile_size, 0))
    if cols[-1] != width - tile_size:
        cols.append(max(width - tile_size, 0))
    return [Window(row=r, col=c, height=min(tile_size, height - r), width=min(tile_size, width - c)) for r in rows for c in cols]


def crop(array: np.ndarray, window: Window) -> np.ndarray:
    return array[..., window.row : window.row + window.height, window.col : window.col + window.width]


def stitch_logits(logit_tiles: list[np.ndarray], windows: list[Window], height: int, width: int) -> np.ndarray:
    if not logit_tiles:
        raise ValueError("No logit tiles to stitch")
    classes = logit_tiles[0].shape[0]
    acc = np.zeros((classes, height, width), dtype="float32")
    count = np.zeros((height, width), dtype="float32")
    for logits, window in zip(logit_tiles, windows):
        acc[:, window.row : window.row + window.height, window.col : window.col + window.width] += logits
        count[window.row : window.row + window.height, window.col : window.col + window.width] += 1
    count[count == 0] = 1
    return acc / count

