from __future__ import annotations

import numpy as np


def build_tiny_random_segformer(num_labels: int = 3):
    """Build a tiny random SegFormer model when torch/transformers are present.

    This helper is intentionally not used by the default workflow because random
    weights would produce meaningless maps. It documents where an open
    checkpoint can be mounted after dataset and license review.
    """
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    config = SegformerConfig(
        num_labels=num_labels,
        num_channels=3,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        decoder_hidden_size=32,
        num_attention_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
    )
    return SegformerForSemanticSegmentation(config)


def prepare_rgb_for_segformer(bands: np.ndarray) -> np.ndarray:
    """Convert a multispectral-like tile to an RGB uint8 image for HF processors."""
    if bands.ndim != 3:
        raise ValueError("bands must have shape (channels, height, width)")
    if bands.shape[0] < 3:
        raise ValueError("at least three bands are required for SegFormer RGB input")
    rgb = np.moveaxis(bands[:3], 0, -1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype("uint8")


def validate_logits_shape(logits: np.ndarray, expected_classes: int, height: int, width: int) -> np.ndarray:
    """Validate and normalize checkpoint logits before stitching."""
    array = np.asarray(logits, dtype="float32")
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("SegFormer logits must have shape (classes, height, width)")
    if array.shape[0] != expected_classes:
        raise ValueError(
            f"Checkpoint returned {array.shape[0]} classes; expected {expected_classes}. "
            "Use a checkpoint trained for this repository's class contract or update the class mapping."
        )
    if array.shape[1:] != (height, width):
        raise ValueError(f"Checkpoint logits have spatial shape {array.shape[1:]}; expected {(height, width)}")
    return array


class SegFormerTilePredictor:
    """Optional Hugging Face SegFormer tile predictor.

    The repository does not bundle weights. This class loads an explicit
    checkpoint supplied by the user and validates that its output class count
    matches the public class contract before the common stitching stage.
    """

    def __init__(self, checkpoint: str, expected_classes: int, local_files_only: bool = False):
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "SegFormer backend requires optional dependencies. "
                "Install with: python -m pip install -e .[segformer]"
            ) from exc

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=local_files_only)
        self._model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, local_files_only=local_files_only)
        self._model.eval()
        self.expected_classes = expected_classes

    def __call__(self, bands: np.ndarray) -> np.ndarray:
        image = prepare_rgb_for_segformer(bands)
        inputs = self._processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = self._torch.nn.functional.interpolate(
                outputs.logits,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
        return validate_logits_shape(
            logits.detach().cpu().numpy(),
            expected_classes=self.expected_classes,
            height=image.shape[0],
            width=image.shape[1],
        )
