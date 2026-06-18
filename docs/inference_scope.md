# Inference Scope

This repository is data-governed by design.

The default workflow does not use:

- restricted imagery;
- restricted labels;
- trained model weights;
- active research metrics;
- commercial satellite products.

The current `v0.1` output is a synthetic coastal scene segmented with a
deterministic lightweight head. It exercises the inference contract:

1. prepare a multispectral-like tile;
2. tile the image;
3. run class-logit inference per tile;
4. stitch logits;
5. export input, mask, overlay and class summary.

It must not be described as a trained SegFormer result.

## Extension path

- Select an open dataset with a clear license.
- Use an open pretrained checkpoint or train a small model on cleared data.
- Add the checkpoint license and dataset citation.
- Keep restricted research data out of the repository.
- State validation metrics only after they are measured on the selected dataset.
