# Model Card: PanelCheck ResNet-18 v2

## Status

| Field | Value |
| --- | --- |
| Task | Four-class, single-label car-body image classification |
| Ordered classes | `normal`, `scratch`, `dent`, `rust` |
| Architecture | ImageNet-pretrained ResNet-18 with four-logit head |
| Framework | PyTorch 2.13.0 / torchvision |
| Checkpoint | `artifacts/v2/models/model.pt` |
| SHA-256 | `e2b329f3d96d1300a6b87bb7b0799d2fee76e12174095de52c19de2243c6ae07` |
| Train / validation / frozen test | 280 / 35 / 35 images |
| Training device | Apple MPS |
| Seed | 42 |
| Selected confidence threshold | 0.65, validation only |

The v2 checkpoint is promoted because its performance on the independently
selected 40-image demonstration set improved from 37.50% to 70.00%. Its v2
frozen-test accuracy is only 71.43%; neither number is a production claim.

## Intended use

The model supports a supervised university demonstration using one clear image
of an exterior car body or panel. Results are advisory and are paired with an
unknown/manual-review state, fixed repair guidance, simulated inventory, and a
human-feedback mechanism.

It must not be used to determine roadworthiness, structural safety, repair cost,
insurance liability, severity, or autonomous purchasing/repair actions. It does
not detect, localize, segment, or support multiple simultaneous defects.

## Input, preprocessing, and outputs

The API accepts one JPEG or PNG up to 10 MB. The complete image is decoded,
EXIF-oriented, converted to RGB, square-padded without cropping, resized to
224×224, converted to a tensor, and normalized with ImageNet mean
`(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`.

The checkpoint stores this preprocessing profile. Evaluation and API inference
read it from metadata. Older checkpoints without the field explicitly fall back
to their legacy direct-stretch resize so baseline reproduction is not silently
changed.

| Logit index | Class |
| ---: | --- |
| 0 | `normal` |
| 1 | `scratch` |
| 2 | `dent` |
| 3 | `rust` |

Class order is checked against `config/class_names.json` when the app starts.

## Training procedure

The model was trained with AdamW, weight decay `1e-4`, batch size 16,
inverse-frequency class weights measured from the training split, and
deterministic seed 42. The 280 training samples were distributed as normal 64,
scratch 80, dent 80, and rust 56; corresponding weights were 1.09375, 0.875,
0.875, and 1.25.

Training used two stages:

1. classification head only: up to 5 epochs at learning rate `1e-3`;
2. complete network: up to 25 epochs at learning rate `5e-5`.

`ReduceLROnPlateau` reduced the learning rate after validation loss stopped
improving. Early stopping patience was six. Training ended at epoch 19 and the
lowest-validation-loss state from epoch 13 (loss 0.1644) was restored. The test
path and test manifest are absent from the training configuration.

Training augmentation uses square padding/resize, horizontal flip, affine
rotation up to 6°, translation up to 2.5%, scale 0.95–1.05, and mild brightness,
contrast, saturation, and hue variation. It does not randomly crop away a
localized defect.

The checkpoint metadata records architecture, classes, image size, complete
preprocessing, seed, timestamps, PyTorch version, device, stages, class
counts/weights, best epoch/loss, threshold provenance, and source/training/
validation manifest hashes.

## Validation and confidence rule

The operational threshold was chosen from 0.30–0.90 using only 35 validation
predictions. At 0.65 it accepted 30/35, rejected 5/35 for manual review, and had
30/30 accepted predictions correct on that validation split. This is a selection
result on a small validation set, not a calibration guarantee.

In the API:

- confidence ≥0.65 returns the most likely supported class;
- confidence <0.65 returns `unknown`, exposes the most likely class separately,
  and withholds repair steps and parts;
- feedback can recall a disclosed label only for the exact same SHA-256 image;
  it does not alter logits or retrain the model.

## Frozen v2 test results

The final test manifest checksum is
`ee358158d8b714f76b963815c12109679d187429a00d258fd35ea6380679f80b`.
It was evaluated once after the checkpoint and threshold were frozen.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `normal` | 0.7778 | 0.8750 | 0.8235 | 8 |
| `scratch` | 0.6429 | 0.9000 | 0.7500 | 10 |
| `dent` | 0.8571 | 0.6000 | 0.7059 | 10 |
| `rust` | 0.6000 | 0.4286 | 0.5000 | 7 |

- Accuracy: **0.7143** (25/35)
- Macro F1: **0.6949**
- Weighted F1: **0.7042**
- Mean confidence, correct: **0.8961**
- Mean confidence, incorrect: **0.7042**
- At threshold 0.65: 29/35 accepted, 6/35 sent to manual review;
  accepted accuracy 22/29 = **0.7586**

The high incorrect-confidence mean shows that 0.65 is not a reliable
out-of-distribution or error detector on the frozen test set.

## External frozen demo results

The 40 demo images were selected and checksummed before any model inference,
with ten images per class and no source-group, exact, or dHash-distance-≤6
overlap with development data.

| Class | Baseline v1 | V2 |
| --- | ---: | ---: |
| `normal` | 10/10 | 9/10 |
| `scratch` | 3/10 | 4/10 |
| `dent` | 2/10 | 5/10 |
| `rust` | 0/10 | 10/10 |
| **Overall** | **15/40 (37.50%)** | **28/40 (70.00%)** |

All twelve failed v2 images remain in the demo folder and results. This external
gain, not the non-comparable internal split percentages, is the evidence for
promotion.

## Comparison with preserved baseline

Baseline v1 used 96 training, 12 validation, and 12 test images. It reached
100% training accuracy from epoch 9 while validation loss remained near 0.97,
evidence of overfitting. Its frozen test accuracy was 75.00%, macro/weighted F1
0.7292, but the same external demo scored only 37.50%.

V2's own frozen-test percentage is lower (71.43%) on a different 35-image split.
It would be academically invalid to call that an improvement. V2's external
demo improvement and expanded traceable training pool are the positive results;
its scratch/dent confusion and weak v2 rust-test recall are unresolved.

## Limitations and risks

- Dataset v2 is still small and remains below 200 images per class.
- Scratch/dent HITL crops and Commons whole-car images differ greatly in scale,
  resolution, background, and source, allowing shortcut learning.
- Many scratch crops are 96 px before upscaling; fine-grained texture may be
  lost.
- The rust demo contains obvious whole-car corrosion while some v2 test rust
  examples are subtle crops, explaining the large 100% vs 42.86% recall gap.
- Scratch and dent are visually overlapping and were frequently confused.
- Single labels exclude real multi-defect complexity.
- Softmax confidence can be high and wrong, especially out of distribution.
- No cross-dataset benchmark, confidence calibration, fairness, adversarial,
  localization, or latency study has been completed.

Users must inspect the original image and defer to a qualified repairer. Even a
high-confidence prediction is only an academic classifier output.
