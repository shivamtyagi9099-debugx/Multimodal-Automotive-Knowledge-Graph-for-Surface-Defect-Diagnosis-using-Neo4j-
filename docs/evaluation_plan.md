# Evaluation Plan and Completed Record

## Isolation protocol

Dataset v2 was visually reviewed before splitting. The accepted 350-image pool
was split 80/10/10 with seed 42. Training used only 280 training and 35
validation images; the 35-image test manifest was checksum-frozen. Architecture,
augmentation, early stopping, and the 0.65 confidence threshold were selected
without test predictions.

The final test was run once after the best-validation-loss checkpoint and
threshold were frozen. The result was not used to retrain, relabel, or tune.
Any future change now requires a new version and a new untouched test set if an
unbiased final estimate is claimed.

## Bound artifacts

| Item | Location or identifier |
| --- | --- |
| Test directory | `data/dataset_v2/processed/test/` |
| Test manifest | `data/dataset_v2/manifests/test_manifest.csv` |
| Test manifest SHA-256 | `ee358158d8b714f76b963815c12109679d187429a00d258fd35ea6380679f80b` |
| Checkpoint | `artifacts/v2/models/model.pt` |
| Checkpoint SHA-256 | `e2b329f3d96d1300a6b87bb7b0799d2fee76e12174095de52c19de2243c6ae07` |
| Evaluation configuration | `ml/configs/evaluation_v2.yaml` |
| Numeric report | `artifacts/v2/metrics/evaluation_report.json` |
| Predictions | `artifacts/v2/metrics/test_predictions.csv` |
| Misclassifications | `artifacts/v2/metrics/misclassification_report.csv` |
| Confidence analysis | `artifacts/v2/metrics/confidence_analysis.json` |
| Confusion matrix | `artifacts/v2/figures/confusion_matrix.png` |

The evaluator verifies the saved test-manifest checksum before loading test
images and records both the model and manifest identifiers in its JSON output.

## Prespecified metrics

For each ordered class, report precision, recall, F1, and support. Also report
overall accuracy, macro F1, weighted F1, the numeric/labelled confusion matrix,
per-image predictions, confidence summaries for correct/incorrect predictions,
and the deployment threshold's coverage/manual-review trade-off.

Argmax predictions are used for four-class metrics. The separate operational
analysis marks confidence below 0.65 for manual review; this preserves a complete
four-class confusion matrix while measuring deployed rejection behavior.

## Completed frozen-test results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `normal` | 0.7778 | 0.8750 | 0.8235 | 8 |
| `scratch` | 0.6429 | 0.9000 | 0.7500 | 10 |
| `dent` | 0.8571 | 0.6000 | 0.7059 | 10 |
| `rust` | 0.6000 | 0.4286 | 0.5000 | 7 |

Accuracy was 25/35 = 0.7143, macro F1 0.6949, and weighted F1 0.7042.
The confusion matrix (true rows, predicted columns in central order) was:

```text
[[7, 0, 0, 1],
 [0, 9, 0, 1],
 [0, 4, 6, 0],
 [2, 1, 1, 3]]
```

Dent→scratch was the largest directional confusion (4/10 dent images). Rust had
the lowest recall (3/7), with errors spread across normal, scratch, and dent.

At 0.65, 29/35 test predictions were accepted and 6/35 sent to manual review.
Accepted accuracy was 22/29 = 0.7586. Mean confidence was 0.8961 for correct and
0.7042 for incorrect predictions, showing that the threshold does not reliably
detect every error outside validation.

## Frozen external demo

The 40 demo images (10/class) were selected from licensed sources, visually
reviewed, and checksummed before model inference. Demo groups were excluded
before v2 acquisition, and the validator confirms no source-group, SHA-256, or
dHash-distance-≤6 overlap with development data.

V2 scored 28/40 = 70.00%: normal 9/10, scratch 4/10, dent 5/10, rust 10/10. The
same demo scored 15/40 = 37.50% with baseline v1. Every failed image remains in
the folder and saved CSV.

## Reproduction

```bash
python scripts/validate_dataset.py \
  --train-manifest data/dataset_v2/manifests/train_manifest.csv \
  --validation-manifest data/dataset_v2/manifests/validation_manifest.csv \
  --test-manifest data/dataset_v2/manifests/test_manifest.csv \
  --test-checksum data/dataset_v2/manifests/test_set_checksum.txt \
  --images-root data/dataset_v2/raw \
  --require-traceability \
  --near-duplicate-distance 6
python scripts/evaluate_model.py --config ml/configs/evaluation_v2.yaml
python scripts/validate_demo_images.py --api-url http://127.0.0.1:8000
```

Repeated execution may reproduce saved results, but no decisions may be revised
from the repeated test/demo outputs. Report all classes and every failure.
