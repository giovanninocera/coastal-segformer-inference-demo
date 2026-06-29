# Changelog

## 0.2.0 - 2026-06-29

- Repositioned the project as a coastal segmentation inference harness rather
  than a bundled trained SegFormer model.
- Added an optional Hugging Face SegFormer backend that loads an explicit
  checkpoint and validates output class shape before stitching.
- Added backend configuration fields and adapter tests that do not require the
  optional ML dependencies.
- Clarified the model boundary in README and inference-scope documentation.

## 0.1.3 - 2026-06-19

- Rebuilt the scene from connected internal- and external-water geometries.
- Preserved the barrier island and lagoon-to-sea openings with west at the top.
- Separated offshore and lagoon reflectance while retaining one water class.
- Added a regression test for lagoon-to-sea connectivity.

## 0.1.2 - 2026-06-19

- Replaced the analytic coastal scene with a generalized lagoon geometry.
- Preserved the source geometry aspect ratio across input, mask and overlay.
- Removed grid artifacts and constrained vegetation to coherent shore zones.
- Renamed the third class to `bare_land` and regenerated all graphics.

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
