# Coastal Segmentation Inference Harness

Geospatial AI inference harness for coastal semantic segmentation, with a
deterministic default backend and an optional Hugging Face SegFormer checkpoint
backend.

The default run uses a procedural multispectral-like coastal scene with
connected open-sea and lagoon zones. A deterministic lightweight segmentation
head exercises preprocessing, tiling, stitching and visualization. The
SegFormer adapter is isolated so that a licensed checkpoint can be mounted
explicitly when available. The public default output is not a trained model
prediction.

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

## Optional SegFormer backend

Install optional dependencies and provide an explicit checkpoint:

```powershell
python -m pip install -e .[segformer]
```

Example config fields:

```yaml
model_backend: segformer
checkpoint: path-or-huggingface-model-id
local_files_only: false
```

The adapter loads the checkpoint through Hugging Face Transformers, upsamples
tile logits to the requested tile size, and validates that the checkpoint class
count matches this repository's `water`, `vegetation`, `bare_land` output
contract before stitching. If the checkpoint uses a different class mapping,
the run fails instead of silently producing misleading maps.
