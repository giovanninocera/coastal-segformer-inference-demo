# Changelog

## 0.1.1 - 2026-06-18

- Improved the synthetic multispectral coastal tile.
- Rebuilt the segmentation triplet with a clean report-style layout and
  separate legend.
- Added lightweight majority filtering to reduce salt-and-pepper class noise in
  the deterministic output.

## 0.1.0 - 2026-06-18

- Added segmentation inference harness.
- Added synthetic multispectral-like tile, tiling, stitching, deterministic
  logits, class summary and input/mask/overlay visualization.
- Added optional Hugging Face SegFormer adapter.
- Added unit tests and GitHub Actions workflow.
