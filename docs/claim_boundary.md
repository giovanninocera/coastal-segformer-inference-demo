# Claim Boundary

This repository is public-safe by design.

The default demo does not use:

- unpublished PhD imagery;
- unpublished labels;
- trained model weights;
- paper-facing metrics;
- commercial satellite products.

The current `v0.1` output is a synthetic coastal scene segmented with a
deterministic lightweight head. It demonstrates the inference contract:

1. prepare a multispectral-like tile;
2. tile the image;
3. run class-logit inference per tile;
4. stitch logits;
5. export input, mask, overlay and class summary.

It must not be described as a trained SegFormer result.

## How to upgrade safely

- Select a public dataset with a clear license.
- Use a public pretrained checkpoint or train a tiny public model.
- Add the checkpoint license and dataset citation.
- Keep PhD and journal-submission data out of the repository.
- State validation metrics only after they are measured on public data.

