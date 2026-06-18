from __future__ import annotations


def build_tiny_random_segformer(num_labels: int = 3):
    """Build a tiny random SegFormer model when torch/transformers are present.

    This helper is intentionally not used by the default demo because random
    weights would produce meaningless maps. It documents where a public
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

