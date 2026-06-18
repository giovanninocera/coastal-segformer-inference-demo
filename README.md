# Coastal SegFormer Inference Demo

Public-safe geospatial AI inference harness for coastal semantic segmentation.

The current `v0.1` is SegFormer-ready but does not publish unpublished PhD data,
labels, model weights or paper-facing metrics. The default run uses a synthetic
multispectral-like tile and a deterministic lightweight segmentation head to
demonstrate preprocessing, tiling, stitching and visualization.

![Segmentation triplet](assets/segmentation_triplet.png)

## Recruiter signal

This repo is meant to show AI-based EO processing without exposing unpublished
PhD assets. It demonstrates the engineering contract around inference:

- multispectral-like input preparation;
- tile/window generation;
- per-tile class logits;
- stitched full-scene prediction;
- input/mask/overlay visualization;
- explicit checkpoint and dataset policy.

## Quick start

```powershell
cd C:\_#\github_portfolio\coastal-segformer-inference-demo
python -m pip install -e .
coastal-segformer-demo run --config configs/example_demo.yml
python -m unittest discover -s tests
```

Without installing:

```powershell
$env:PYTHONPATH = "src"
python -m coastal_segformer_demo run --config configs/example_demo.yml
python -m unittest discover -s tests
```

## Outputs

| Output | Purpose |
|---|---|
| `outputs/demo/synthetic_rgb.png` | Public-safe synthetic input |
| `outputs/demo/segmentation_triplet.png` | Input, predicted classes and overlay |
| `outputs/demo/class_summary.csv` | Class fractions |
| `outputs/demo/metadata.json` | Claim boundary and run metadata |

## Important limitation

This repository must not publish unpublished `_ST` data, labels, trained weights
or paper-facing results. It can demonstrate the inference pattern and the
geospatial handling around a model, but not leak the thesis dataset.

The current default output must not be described as a trained SegFormer result.
See `docs/claim_boundary.md`.

## Optional SegFormer adapter

`src/coastal_segformer_demo/segformer_adapter.py` shows where a Hugging Face
SegFormer checkpoint can be mounted after public dataset and checkpoint-license
review. It is not used by the default run because random weights would produce
meaningless maps.

## Public-safe upgrade options

| Option | Pros | Cons |
|---|---|---|
| Public benchmark dataset | Real data and clearer reproducibility | Need dataset-license check |
| Synthetic tile fixture | Fully safe and fast | Less impressive visually |
| Pretrained generic SegFormer | Easy to run | May not be EO-specific |
| Tiny trained model on public data | Best demo if clean | More work and validation burden |

## Current roadmap

- [x] Synthetic multispectral-like fixture.
- [x] Tiling and stitching.
- [x] Class-logit inference contract.
- [x] Input/mask/overlay figure.
- [x] Tests.
- [ ] Choose a public segmentation dataset.
- [ ] Add a real public checkpoint or train a tiny public model.
- [ ] Add measured public-data metrics.

## Suggested GitHub topics

`segformer`, `semantic-segmentation`, `earth-observation`, `geospatial-ai`,
`pytorch`, `transformers`, `remote-sensing`
