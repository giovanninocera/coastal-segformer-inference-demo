# Inference Scope

This repository is data-governed by design.

The default workflow does not use:

- restricted imagery;
- restricted labels;
- trained model weights;
- active research metrics;
- commercial satellite products.

The default output is a synthetic coastal scene segmented with a
deterministic lightweight head. It exercises the inference contract:

1. prepare a multispectral-like tile;
2. tile the image;
3. run class-logit inference per tile;
4. stitch logits;
5. export input, mask, overlay and class summary.

It must not be described as a trained SegFormer result.

## Backend contract

The default backend is `heuristic`. It is deterministic and exists to test the
engineering path around inference.

The optional backend is `segformer`. It requires:

- `python -m pip install -e .[segformer]`;
- an explicit `checkpoint` in the YAML config;
- a checkpoint whose output class count matches the public class contract:
  `water`, `vegetation`, `bare_land`.

The adapter validates checkpoint logits before stitching. A generic SegFormer
checkpoint trained on unrelated classes should not be used as evidence of
coastal segmentation performance.

## Extension path

- Select an open dataset with a clear license.
- Use an open pretrained checkpoint or train a small model on cleared data.
- Add the checkpoint license and dataset citation.
- Keep restricted research data out of the repository.
- State validation metrics only after they are measured on the selected dataset.
