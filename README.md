# PanelCheck: Car-Body Panel Defect Classification

PanelCheck is a professor-ready academic proof-of-concept that classifies one
car-body image into exactly four ordered classes: `normal`, `scratch`, `dent`,
and `rust`. A PyTorch ResNet-18 model is served by FastAPI, while Neo4j supplies
fixed repair guidance and simulated part/inventory relationships. The same
FastAPI process serves the browser interface.

This is not a safety, roadworthiness, repair-cost, or production diagnostic
system. Repair guidance is fixed educational content. Every price, stock level,
and inventory date is simulated.

## Verified status

The promoted v2 checkpoint is `artifacts/v2/models/model.pt`. Its test and demo
results are deliberately reported without cherry-picking:

| Evidence | Baseline v1 | Promoted v2 |
| --- | ---: | ---: |
| Development images | 120 | 350 human-reviewed |
| Frozen test images | 12 | 35 |
| Frozen-test accuracy | 75.00% | 71.43% |
| Frozen-test macro F1 | 0.7292 | 0.6949 |
| Honest external demo accuracy | 37.50% | 70.00% |

The two internal test scores are not directly comparable because the v2 test
set is a new, larger frozen split. The external 40-image demo is the same
preselected set for both checkpoints, so its 32.5 percentage-point gain is the
clearest generalization improvement. V2 is still weak at separating subtle
scratches from dents; see `docs/model_card.md` and
`docs/baseline_audit.md` before presenting the results.

## Professor Demonstration

From the project directory, activate the existing environment. Install packages
only if the environment is new or incomplete:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create the local secret file once and replace the placeholder password with a
new local password. Never commit `.env`.

```bash
cp -n .env.example .env
```

If `.env` already exists, keep it; `cp -n` will not overwrite the local
password. Edit only a newly created file.

Start, seed, and verify Neo4j:

```bash
docker compose up -d --wait neo4j
python scripts/seed_neo4j.py
python scripts/check_neo4j.py
```

Start the application:

```bash
python scripts/run_api.py --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). In a second terminal,
run the frozen demo set through the real API:

```bash
source .venv/bin/activate
python scripts/validate_demo_images.py --api-url http://127.0.0.1:8000
```

The expected honest result for the promoted checkpoint is 28/40, or 70.00%:
normal 9/10, scratch 4/10, dent 5/10, and rust 10/10. Do not remove the failed
images or present only successful examples.

To reproduce the already-frozen final evaluation:

```bash
python scripts/evaluate_model.py --config ml/configs/evaluation_v2.yaml
```

Stop FastAPI with `Control-C`, then stop Neo4j:

```bash
docker compose stop
```

If port 8000 is occupied, the launcher now stops with a clear message. Find the
owner or choose a different port:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
python scripts/run_api.py --port 8899
```

Use the same alternate base URL for the browser and demo validator.

## What the application verifies

- JPEG/PNG uploads are size-checked and fully decoded before inference.
- EXIF orientation is applied, images are converted to RGB, square-padded,
  resized to 224×224, and normalized with the same checkpoint-defined profile.
- Confidence below the validation-selected threshold of `0.65` returns
  `unknown` and withholds repair/part recommendations.
- Known predictions use Neo4j when connected and a clearly labelled static
  catalog fallback when it is unavailable.
- Feedback is appended locally and may recall a disclosed correction only for
  the exact same SHA-256 image bytes. It never retrains the model.
- The local favicon prevents irrelevant missing-favicon errors during a demo.

## Dataset v2

V2 contains 350 accepted images: normal 80, scratch 100, dent 100, and rust 70.
It uses the public Humans in the Loop Car Parts and Car Damages dataset at a
pinned mirror revision under CC0-1.0, plus individually licensed Wikimedia
Commons images from the `Quality images of automobiles` and `Rusty automobiles`
categories. Exact source pages and licences are recorded per image.

Candidate selection was completed before model inference. Contact sheets were
reviewed against written rules, 147 of 497 candidates were rejected, related
sequences were reduced to one representative, and the accepted pool was split
80/10/10 with seed 42. SHA-256 and 64-bit difference hashes are validated across
splits and against the demo set. The final test manifest is checksum-frozen.

The complete reproducible preparation sequence is:

```bash
python scripts/acquire_dataset_v2_candidates.py
python scripts/acquire_v2_rust_commons.py
python scripts/create_v2_review_sheets.py
# Inspect the contact sheets and freeze the decisions before inference.
python scripts/apply_v2_review_decisions.py
python scripts/prepare_dataset_v2.py
python scripts/validate_dataset.py \
  --train-manifest data/dataset_v2/manifests/train_manifest.csv \
  --validation-manifest data/dataset_v2/manifests/validation_manifest.csv \
  --test-manifest data/dataset_v2/manifests/test_manifest.csv \
  --test-checksum data/dataset_v2/manifests/test_set_checksum.txt \
  --images-root data/dataset_v2/raw \
  --require-traceability \
  --near-duplicate-distance 6
```

Acquisition scripts refuse to overwrite an existing candidate pool. The checked
in `source_records.csv` and `review_decisions.json` are the audit record for this
version; create a new versioned directory for a different review rather than
silently replacing v2. See `docs/dataset_card.md` for licence and exclusion
details.

## Training design

The v2 experiment uses ImageNet-pretrained ResNet-18, AdamW, deterministic seed
42, measured inverse-frequency class weights, early stopping, and
`ReduceLROnPlateau`. It trains the classification head for up to five epochs,
then unfreezes the full network at a lower learning rate for up to 25 more. The
best validation-loss state was epoch 13; early stopping ended at epoch 19.

Augmentation preserves defect evidence: square pad and resize, horizontal flip,
small affine rotation/translation/scale, and mild colour variation. Validation,
test, API, and demo inference are deterministic. Threshold selection uses only
validation predictions. Training code has no test-path input.

To train a new version after a newly reviewed and frozen dataset is prepared:

```bash
python scripts/train_model.py --config ml/configs/training_v2.yaml
```

Do not tune after reading `artifacts/v2/metrics/evaluation_report.json`; doing so
would invalidate the frozen test claim.

## Verification commands

```bash
pytest
python scripts/check_neo4j.py
python scripts/verify_live_api.py --api-url http://127.0.0.1:8000
python scripts/validate_demo_images.py --api-url http://127.0.0.1:8000
```

The completed verification run produced 54 passing tests. Live checks covered
`GET /`, `GET /health`, `GET /openapi.json`, `POST /api/v1/predict`, and
`POST /api/v1/feedback`; they also confirmed recovery from Neo4j to the static
fallback and back.

## Important files

- `docs/baseline_audit.md` — evidence-based diagnosis of the original model.
- `docs/dataset_card.md` — v2 sources, licences, counts, review, and exclusions.
- `docs/model_card.md` — checkpoint metadata, metrics, confidence, and limits.
- `demo_images/` — frozen 40-image demonstration set and genuine results.
- `artifacts/baseline_v1/` — preserved baseline checkpoint and evidence.
- `artifacts/v2/` — promoted checkpoint, metrics, figures, and live checks.
- `artifacts/v2/metrics/comparison_summary.json` — machine-readable result
  summary, including which comparisons are and are not valid.

## Remaining limitations

- The task assigns one image-level class; it does not locate damage or support
  multiple defects.
- V2 remains far below the aspirational 200 images per class because suitable
  licensed, visually unambiguous examples were not available in the selected
  sources. Quality was not relaxed to reach a number.
- Scratch crops are often much lower resolution than normal/rust whole-car
  images, leaving measurable scale and source shortcuts.
- The 35-image test and 40-image demo are too small for production, fairness,
  calibration, or safety claims.
- High confidence can still be wrong. The threshold is a manual-review rule,
  not a calibrated probability or out-of-distribution detector.
- Repair steps are general educational guidance, and all inventory data is
  simulated.
# Multimodal-Automotive-Knowledge-Graph-for-Surface-Defect-Diagnosis-using-Neo4j-
