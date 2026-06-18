from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import InferenceConfig
from .predictor import CLASS_NAMES, heuristic_logits, logits_to_mask
from .render import save_triplet
from .synthetic import make_synthetic_multispectral
from .tiling import crop, make_windows, stitch_logits


@dataclass(frozen=True)
class InferenceResult:
    output_dir: Path
    artifacts: list[Path]


def run_inference(config: InferenceConfig) -> InferenceResult:
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tile = make_synthetic_multispectral(config.width, config.height, config.seed)
    windows = make_windows(config.height, config.width, config.tile_size, config.stride)
    logits_tiles = [heuristic_logits(crop(tile.bands, window)) for window in windows]
    logits = stitch_logits(logits_tiles, windows, config.height, config.width)
    mask = logits_to_mask(logits)

    input_path = out / "synthetic_rgb.png"
    Image.fromarray(tile.rgb, mode="RGB").save(input_path)

    triplet_path = out / "segmentation_triplet.png"
    save_triplet(triplet_path, tile.rgb, mask)

    summary_path = out / "class_summary.csv"
    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        count = int((mask == class_id).sum())
        rows.append({"class_id": class_id, "class_name": name, "pixels": count, "fraction": count / mask.size})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "pixels", "fraction"])
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = out / "metadata.json"
    metadata = {
        "title": "Coastal segmentation inference",
        "data_scope": "synthetic fixture",
        "processing_note": "Deterministic head; not a trained SegFormer result.",
        "tile_size": config.tile_size,
        "stride": config.stride,
        "windows": len(windows),
        "classes": CLASS_NAMES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return InferenceResult(out, [input_path, triplet_path, summary_path, metadata_path])
