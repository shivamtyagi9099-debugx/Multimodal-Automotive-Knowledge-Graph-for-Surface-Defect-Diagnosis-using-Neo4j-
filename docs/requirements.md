# Software Requirements Specification

## 1. Purpose and Status

This document defines the approved requirements for a single-semester academic
proof-of-concept (PoC) that classifies images of car-body panels. The software,
reviewed dataset v2, model training, frozen evaluation, and live application
verification are implemented. Results remain limited academic evidence.

## 2. Scope

The system accepts one image of a car-body panel and classifies the complete
image into one of four ordered classes:

1. `normal`
2. `scratch`
3. `dent`
4. `rust`

This is image classification only. The system does not locate a defect, draw a
bounding box, produce a segmentation mask, search for similar images, generate
repair text, or query live suppliers.

## 3. Functional Requirements

| ID | Requirement |
| --- | --- |
| FR-01 | The UI shall accept one JPEG or PNG image no larger than 10 MB. |
| FR-02 | The backend shall verify the declared media type, filename extension, file size, and decodability before saving an upload. |
| FR-03 | The classifier shall produce a probability distribution over the four centrally configured classes. |
| FR-04 | A confidence at or above the checkpoint's validation-selected threshold (`0.65` for v2) shall return the most likely supported class. |
| FR-05 | A confidence below that threshold shall return `Defect unknown - Please review manually.` and shall not return repair or inventory recommendations. |
| FR-06 | A known prediction shall be enriched with fixed repair guidance from Neo4j when connected or the clearly labelled local static fallback when unavailable. No generative model shall create repair instructions. |
| FR-07 | A known prediction shall return simulated part identifiers, names, INR prices, fake stock quantities, availability states, and a visible static last-updated date. |
| FR-08 | The UI shall state that inventory data is simulated and that the application is not a universal diagnostic tool. |
| FR-09 | The UI shall ask `Was this prediction correct?`, require a corrected class for a No response, and support an optional comment. |
| FR-10 | Feedback shall be appended to a local CSV file with a UTC timestamp and prediction identifier. |
| FR-11 | The API shall expose health, prediction, and feedback routes and report model/catalog/Neo4j readiness plus the active knowledge source. |
| FR-12 | The offline evaluation shall produce precision, recall, F1-score, and support for every class, a four-class confusion matrix, and secondary overall accuracy. |
| FR-13 | Dataset preparation shall create a reproducible, class-stratified `80/10/10` train/validation/test split. |
| FR-14 | The test manifest shall be protected by a SHA-256 checksum and remain unused until final evaluation. |

## 4. Non-Functional Requirements

| Area | Requirement |
| --- | --- |
| Reproducibility | Dataset splitting and training shall use the configured random seed `42`; checkpoint metadata shall record the class order and training settings required for inference. |
| Data integrity | Duplicate image identifiers, paths, or checksums shall not appear across train, validation, and test manifests. |
| Transparency | Unknown predictions, missing checkpoints, simulated inventory, data dates, limitations, and incomplete evaluation evidence shall be shown honestly. |
| Privacy | Uploaded images and feedback shall remain local and shall not be committed to version control. Uploads shall not become training data without review, consent, and formal manifest inclusion. |
| Maintainability | Taxonomy, application settings, ML settings, static guidance, and code shall remain separated in their existing configuration and package files. |
| Usability | The frontend shall support keyboard focus, clear validation errors, loading states, image preview, responsive layouts, and reduced-motion preferences. |
| Safety | Repair guidance shall remain conservative, source-backed, and subordinate to vehicle-maker procedures and qualified professional inspection. |
| Portability | The application shall run locally through Python, PyTorch, FastAPI, and a browser without requiring a cloud service or live external API. |

No latency or accuracy target is claimed before the hardware environment and
selected dataset are documented. Measured values must be reported rather than
retrospectively selecting a target that matches the result.

## 5. Class Definitions

| Class | Operational definition | Important boundary |
| --- | --- | --- |
| `normal` | The visible panel surface contains none of the three supported defect types. | Dirt, glare, reflections, panel seams, and design features must not be labelled as defects without supporting evidence. |
| `scratch` | A visible linear or localized abrasion affecting the surface finish or paint layers. | The class does not encode scratch depth, length, or repair severity. |
| `dent` | A visible inward or outward deformation of the panel contour. | The class does not encode dent dimensions, structural damage, or paint condition. |
| `rust` | Visible oxidation or corrosion on the panel surface. | The class does not determine hidden corrosion, perforation, or structural safety. |

If an image contains multiple supported defects, the dataset must define a
single, documented label policy before it is admitted. Multi-label
classification is outside the current scope.

## 6. Confidence-Threshold Contract

- Let `p_max` be the largest softmax probability.
- When `p_max >= 0.65`, return the corresponding configured class and its
  catalog entry.
- When `p_max < 0.65`, set `predicted_class` to `unknown`, retain the closest
  class only for transparency, require manual review, and return the exact
  unknown message.
- The threshold is a review rule, not proof that a prediction is correct.
- Threshold suitability must be examined on validation data and reported
  without altering the untouched test set.

## 7. Static Catalog and Inventory

`data/static/defect_catalog.json` is the reproducible seed and fallback source
for display names, repair steps, and mock parts. Neo4j is the preferred live
query source. The `normal` class returns no repair action or
replacement parts. Inventory identifiers begin with `SIM-`, prices and stock
are fictional, and the current displayed date is `2026-08-19`.

Repair sequences are conservative paraphrases of manufacturer collision-repair
guidance, including [3M paint-finishing procedures](https://www.3m.com/3M/en_US/collision-repair-us/applications/paint-finishing-and-detail-shop/),
[small-damage repair procedures](https://www.3m.com/3M/en_US/collision-repair-us/applications/metal-shop/small-damage-repair-sop/),
and [corrosion-protection procedures](https://www.3m.com/3M/en_US/collision-repair-us/applications/corrosion-protection/).
They must be reviewed with the project supervisor before the final demonstration.

## 8. Feedback Contract

Each accepted feedback row contains:

- UTC timestamp;
- prediction identifier;
- original image filename;
- exact-image SHA-256 fingerprint when the feedback follows a prediction;
- predicted class and confidence;
- `is_correct` Boolean;
- optional corrected class from the approved taxonomy; and
- optional comment of at most 500 characters.

Feedback is evaluation evidence for later analysis. It shall not trigger
automatic retraining or silently change the current model. A valid corrected
label may be recalled for the exact same image fingerprint only when the
prediction response clearly discloses that feedback was applied.

## 9. Academic Disclaimer

This application is an academic proof-of-concept for a restricted image
classification task. It is not a universal vehicle diagnostic tool, does not
assess roadworthiness or structural safety, and is not a substitute for a
qualified inspection or the vehicle manufacturer's repair procedure.

## 10. Acceptance Criteria

| Criterion | Verification | Current state |
| --- | --- | --- |
| Valid JPEG/PNG upload is accepted and safely renamed | Backend and end-to-end tests | Implemented |
| Invalid type or unreadable image is rejected | Backend tests | Implemented |
| Known prediction includes class, confidence, guidance, inventory, date, and disclaimer | Backend and end-to-end tests | Implemented with deterministic test model |
| Confidence below `0.65` returns the exact unknown rule and no guidance or parts | Backend test | Implemented |
| Y/N feedback and corrections append to local CSV | Backend and end-to-end tests | Implemented |
| Frontend is responsive and exposes transparent model readiness | Static and browser validation | Implemented |
| Strict `80/10/10` split is reproducible and exact/near-duplicate checks pass | ML tests and generated manifests | Implemented for 350 reviewed images |
| A real checkpoint is trained without accessing the test set | Training history and checkpoint metadata | Implemented |
| Final per-class metrics and confusion matrix are generated from the frozen test set | Evaluation artifacts and matching checksum | Implemented |

## 11. Explicitly Out of Scope

- object detection and segmentation;
- similar-image or reverse-image search;
- multi-object or universal defect diagnosis;
- AI-generated repair advice;
- live inventory, supplier, pricing, or purchasing APIs;
- authentication, multi-user accounts, and production databases;
- automated retraining; and
- production or safety-critical deployment.
