# Coastal SegFormer Inference

Geospatial AI inference harness for coastal semantic segmentation.

The default run uses a procedural multispectral-like tile and a deterministic
lightweight segmentation head to exercise preprocessing, tiling, stitching and
visualization. The SegFormer adapter is isolated so that a licensed checkpoint
can be mounted explicitly when available.

![Segmentation triplet](assets/segmentation_triplet.png)

## Scope

The harness covers the engineering contract around coastal segmentation
inference:

- multispectral-like input preparation;
- tile/window generation;
- per-tile class logits;
- stitched full-scene prediction;
- input/mask/overlay visualization;
- explicit checkpoint loading boundary.

## Quick start

```powershell
cd coastal-segformer-inference
python -m pip install -e .
coastal-segformer-inference run --config configs/example_inference.yml
python -m unittest discover -s tests
```

Without installing:

```powershell
$env:PYTHONPATH = "src"
python -m coastal_segformer_inference run --config configs/example_inference.yml
python -m unittest discover -s tests
```

## Outputs

| Output | Purpose |
|---|---|
| `outputs/inference/synthetic_rgb.png` | Synthetic input |
| `outputs/inference/segmentation_triplet.png` | Input, predicted classes and overlay |
| `outputs/inference/class_summary.csv` | Class fractions |
| `outputs/inference/metadata.json` | Processing scope and run metadata |

## Model boundary

The repository does not distribute a model checkpoint. The deterministic
reference head keeps preprocessing, tiling, stitching and output contracts
reproducible; it is not a trained SegFormer result. See
`docs/inference_scope.md`.

## Optional SegFormer adapter

`src/coastal_segformer_inference/segformer_adapter.py` defines the Hugging Face
SegFormer integration boundary and validates checkpoint output shape before
the common stitching and rendering stages.
