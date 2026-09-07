# Dataset Card: PanelCheck v2

## Status and intended use

PanelCheck v2 is a human-reviewed, traceable academic dataset for single-label
classification of a visible car-body panel as one of four ordered classes:

1. `normal`
2. `scratch`
3. `dent`
4. `rust`

It is intended only for this university proof-of-concept. It is not suitable
for roadworthiness, structural, insurance, cost, severity, detection,
segmentation, or production claims.

## Version and measured counts

Acquisition produced 497 candidates. Contact-sheet review accepted 350 and
rejected 147. No count was increased by synthetic or unlicensed material.

| Class | Candidates | Rejected | Accepted | Train | Validation | Frozen test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `normal` | 130 | 50 | 80 | 64 | 8 | 8 |
| `scratch` | 120 | 20 | 100 | 80 | 10 | 10 |
| `dent` | 120 | 20 | 100 | 80 | 10 | 10 |
| `rust` | 127 | 57 | 70 | 56 | 7 | 7 |
| **Total** | **497** | **147** | **350** | **280** | **35** | **35** |

The requested 200 images per class was an aspiration, not a quota. The selected
licensed sources did not contain 200 genuinely distinct, clear examples for
each target—particularly body-panel corrosion—so quality rules were not relaxed.

## Sources and licences

### Humans in the Loop Car Parts and Car Damages Dataset

- Canonical publisher: Humans in the Loop.
- Canonical page:
  `https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/`
- Pinned acquisition mirror: `DrBimmer/car-parts-and-damage-dataset`, revision
  `2dbf46f7e23eb0d6f551d38e4f77a11fe9d045b9`.
- Licence: CC0 1.0, as stated by the canonical publisher.
- Accepted records: 100 scratch, 100 dent, and 16 rust.
- Mapping: `Scratch` → `scratch`, `Dent` → `dent`, `Corrosion` → `rust`.

The mirror advertises an MIT licence, while the canonical publisher identifies
the dataset as CC0 1.0. V2 records the canonical publisher's CC0 designation and
retains the exact pinned mirror URL used to retrieve every file. The old v1
manifest is preserved unchanged with its original MIT record so this provenance
correction remains auditable rather than rewritten retrospectively.

### Wikimedia Commons

- Category `Quality images of automobiles`: 80 accepted normal images.
- Category `Rusty automobiles`: 54 accepted rust images.
- Licence: recorded per file from Commons `extmetadata`, including CC0,
  CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0, and GFDL 1.2 where applicable.
- Attribution: each source page, licence URL, and creator string is retained in
  `data/dataset_v2/source_records.csv`.

Commons category membership is only a candidate-discovery mechanism. A human
visual review, not category membership or model output, determined inclusion.

## Review and class rules

The review protocol was written and applied before v2 model inference. Every
candidate appeared on a contact sheet under `artifacts/v2/review_sheets/`.
Decisions are recorded in `data/dataset_v2/source_records.csv` and the accepted
class indices/rules in `data/dataset_v2/review_decisions.json`.

An accepted image had to:

- visibly contain a car-body panel;
- have one unambiguous target class;
- be a clearly undamaged exterior for `normal`;
- show visible abrasion for `scratch`, contour deformation for `dent`, or
  oxidation/corrosion on a body panel for `rust`;
- decode correctly and have usable visual resolution; and
- be free of dominant watermarks, diagrammatic content, or heavy editing.

Images were rejected for ambiguity, multiple equally dominant defects,
unrelated content, tiny/distant vehicles, wheel-only rust, severe obstruction,
extreme blur/low resolution, heavy editing, or membership in an obvious related
sequence when another representative was retained.

The v1 mistake—treating a body-part annotation as proof of no damage—is not used
in v2. Commons normal candidates have the source description “quality automobile
photograph; requires no-damage review,” and `normal` is assigned only after the
visual decision is accepted.

## Provenance and manifest schema

The accepted source manifest and every split manifest contain:

```text
image_id
file_path
class_name
original_label
source_dataset
source_url
licence
checksum
perceptual_hash
dataset_split
```

They also retain `source_group`, `review_status`, and `review_notes`. SHA-256
binds exact bytes; a 64-bit difference hash supports near-duplicate review.
Accepted file paths and image IDs are stable and unique.

## Split and leakage controls

`scripts/prepare_dataset_v2.py` creates a deterministic, class-stratified
80/10/10 split with seed 42. The test manifest SHA-256 is stored in
`data/dataset_v2/manifests/test_set_checksum.txt` and is checked before any
evaluation.

The preparation/validation gates reject:

- duplicate image IDs, paths, checksums, or source groups across splits;
- invalid or missing provenance values;
- missing, undecodable, checksum-mismatched, or perceptual-hash-mismatched files;
- any cross-split dHash pair at Hamming distance 6 or less; and
- any class order/distribution inconsistent with the central taxonomy.

The accepted pool has zero exact duplicates and zero dHash pairs at distance ≤6.
The frozen demo validator separately confirms zero source-group, checksum, or
dHash-distance-≤6 overlap between all 40 demo images and v2 development data.
A dHash threshold is a conservative review signal, not proof that more distant
images are unrelated.

## Image handling

Source images are decoded with Pillow, EXIF orientation is applied, and pixels
are converted to RGB. HITL defect polygons are expanded to square crops with
context; Commons images remain whole photographs. Training then uses square
padding and resize rather than an evidence-concealing random crop. Validation,
test, demo, and application inference use the same deterministic geometry and
ImageNet normalization stored in the checkpoint.

Measured short-side medians are 853 px for normal, 96 px for scratch, 166.5 px
for dent, and 853 px for rust. Sixty-eight scratch images, 14 dent images, and
11 rust images are below 128 px. This source/scale disparity is a material
remaining shortcut risk and is documented rather than hidden.

## Ethical considerations and limitations

- Public-source imagery may encode geography, vehicle-age, colour, camera,
  socioeconomic, and web-curation biases.
- Commons normal and rust images are often whole vehicles, while HITL scratch
  and dent records are annotated crops; background and scale can reveal source.
- Rust examples may overrepresent old or abandoned vehicles and should not be
  interpreted as normal ownership or maintenance patterns.
- Single-label review necessarily excludes or simplifies multi-defect reality.
- The 35-image test is too small for production, safety, fairness, or subgroup
  conclusions.
- Creators and licences must remain attached if Commons images are redistributed;
  consult each recorded source page for its exact terms.
- No person-identification or sensitive attribute is an intended label. Images
  should not be repurposed for surveillance or identity inference.

## Reproduction gate

Training is authorized only after the review statuses are frozen, preparation
completes, the strict validator passes, and the final test checksum is recorded.
Threshold and hyperparameter choices may use training/validation data only.
Presentation-demo images are excluded before candidate selection and are never
moved into development data after an error.
