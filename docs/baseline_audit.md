# Baseline v1 Audit

This audit preserves the original checkpoint and evidence under
`artifacts/baseline_v1/`. No baseline test image or result was changed while v2
was developed.

## Reproduced baseline

- Dataset: 120 images, exactly 30 per class.
- Split: 96 train, 12 validation, 12 frozen test.
- Checkpoint SHA-256:
  `070b774d1479b9bb38562705c423277a36c9c3327ee376a9082f13f9751a3418`.
- Frozen-test accuracy: 9/12 = 75.00%.
- Macro and weighted F1: 0.7292.
- Honest preselected demo: 15/40 = 37.50%.
- Tests before the changes: 47 passed.

| Class | Precision | Recall | F1 | Test support | Demo correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| `normal` | 1.000 | 1.000 | 1.000 | 3 | 10/10 |
| `scratch` | 1.000 | 0.333 | 0.500 | 3 | 3/10 |
| `dent` | 0.667 | 0.667 | 0.667 | 3 | 2/10 |
| `rust` | 0.600 | 1.000 | 0.750 | 3 | 0/10 |

Confidence on the 12 test images ranged from 0.4247 to 0.9874, with median
0.8724. Mean confidence was 0.8657 for correct predictions and 0.6009 for
incorrect predictions. One incorrect prediction still had confidence 0.7534,
so the old 0.60 threshold did not reliably isolate errors.

## Evidence by suspected cause

| Suspected issue | Evidence and conclusion |
| --- | --- |
| About 30 images/class | Confirmed. Only 24 training images/class and 3 validation/test images/class made metrics high-variance and memorization easy. |
| Incorrect `normal` labels | Confirmed. All 30 normal records originated from body-part labels such as `Front-bumper`, `Front-door`, `Hood`, and `Quarter-panel`; zero had an explicit `normal`, `undamaged`, or `no damage` source label. Visual contact-sheet review found visibly damaged examples. A part annotation had been incorrectly treated as proof of no damage. |
| Ambiguous or multi-defect labels | Confirmed by visual review, particularly scratched/dented collision examples and corrosion on wheels rather than body panels. |
| Class imbalance | Not present in v1: all four classes had exactly 30 records and 24 training samples. Class weights would not address its main failure. |
| Background/source shortcuts | Strongly indicated. Median crop short side was 473 px for normal but only 96 for scratch, 109 for dent, and 106.5 for rust. Normal examples were broader body-part crops while damage examples were tight low-resolution crops. |
| Exact/near duplicate leakage | No evidence at the implemented threshold. SHA/file/source identities did not cross splits, and the audit found zero 64-bit dHash pairs at Hamming distance ≤6 among 120 images. This does not exclude every possible semantic or sequence relation. |
| Preprocessing mismatch | Evaluation and inference both used direct 224×224 resize and ImageNet normalization, but the checkpoint did not record its geometry. Training used random resized crops that could remove a localized defect; direct resize also distorted aspect ratio. Compatibility depended on code defaults rather than checkpoint metadata. |
| Augmentation | The random resized crop (scale 0.8–1.0) was unnecessarily risky for small localized scratches. Rotation/jitter were modest. V2 removes random cropping and uses small evidence-preserving affine/colour changes. |
| Insufficient/unstable fine-tuning | The old run fully fine-tuned ResNet-18 from epoch 1 on only 96 images. Training accuracy reached 100% by epoch 9 while best validation loss remained 0.9723 and early stopping occurred at epoch 14: clear overfitting. |
| Confidence threshold | The 0.60 threshold was a configured constant. It accepted two errors among nine accepted test predictions and was not selected through a documented validation-only objective. |
| Orientation/colour | All audited files had default EXIF orientation, so orientation did not explain this particular test failure. The old loader converted to RGB, but did not consistently apply EXIF orientation first. V2 now applies orientation and RGB conversion in acquisition, training, evaluation, and API inference. |

## Root diagnosis

The apparent 75% baseline result came from only 12 in-source test crops. The
37.5% result on a frozen external set demonstrated that the model had learned a
small, biased source distribution rather than robust defect concepts. The most
serious integrity fault was the normal-class construction; the strongest
technical risks were tiny sample size, class-dependent scale/source cues, and
full-network overfitting from the first epoch.

## Remediation implemented

- Preserved the complete v1 checkpoint, metrics, confusion matrix, confidence
  report, misclassification report, and demo results.
- Added visual-review contact sheets and an evidence-generating dataset audit.
- Built a 350-image v2 pool with explicit accepted/rejected decisions and
  corrected `normal` ground truth.
- Added full per-image provenance, SHA-256, dHash, split/source grouping, and
  frozen-test validation.
- Made EXIF/RGB/geometry preprocessing checkpoint-defined and compatible across
  training, evaluation, and inference.
- Replaced risky random crops with square padding and mild affine/colour
  augmentation.
- Added staged head/full fine-tuning, measured class weights, scheduler, early
  stopping, and validation-only threshold selection.
- Added a frozen external demo whose errors are retained, not fed back into
  training, and evaluated through the real API.
