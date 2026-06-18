# Release Checklist

- [ ] `python -m unittest discover -s tests` passes.
- [ ] Inference run writes input image, segmentation triplet, class summary and metadata.
- [ ] README image renders on GitHub.
- [ ] Scope notes state that default output is not a trained SegFormer result.
- [ ] No model weights are committed.
- [ ] No restricted data or active research metrics are included.
- [ ] GitHub Actions pass after push.
