# System Architecture

## 1. Architecture Status

The frontend, FastAPI routes, static catalog, feedback persistence and recall,
PyTorch pipeline, trained checkpoint, evaluation artifacts, tests, and
command-line launchers are implemented.

## 2. High-Level Runtime Flow

```text
Browser
  |
  | GET / and /static/*
  v
FastAPI application
  |
  +-- GET /health ----------------------> readiness response
  |
  +-- POST /api/v1/predict
  |      |
  |      +--> validate and safely store image
  |      +--> deterministic preprocessing
  |      +--> PyTorch ResNet-18 inference
  |      +--> apply checkpoint threshold (0.65 for v2)
  |      +--> recall disclosed correction for exact image SHA-256
  |      +--> Neo4j repair/inventory lookup
  |      +--> labelled static-catalog fallback when Neo4j is unavailable
  |      `--> transparent prediction response
  |
  `-- POST /api/v1/feedback ------------> local CSV correction history
```

The frontend and API share one origin. No CORS configuration, external
database, cloud service, generative model, or supplier API is required.

## 3. Offline ML Flow

```text
Public source images + source metadata
              |
              v
      dataset_manifest.csv
              |
       validate and split
              |
      +-------+--------+
      |       |        |
    train  validation  test (frozen)
      |       |        |
      +--- training    +--> SHA-256 protected manifest
              |
          model.pt
              |
       final evaluation only
              |
      metrics + confusion matrix
```

Training uses only `data/dataset_v2/processed/train` and
`data/dataset_v2/processed/validation`. The test set is referenced exclusively by the
final evaluation workflow after checksum verification.

## 4. Component Map

| Component | Primary location | Responsibility |
| --- | --- | --- |
| Browser UI | `frontend/` | Upload preview, readiness display, results, simulated inventory, disclaimer, and feedback controls. |
| Application factory | `backend/app/main.py` | Load configuration, class order, catalog, optional checkpoint, routes, and static frontend. |
| Prediction route | `backend/app/api/routes/prediction.py` | Orchestrate validation, storage, preprocessing, inference, thresholding, and response enrichment. |
| Feedback route | `backend/app/api/routes/feedback.py` | Validate supported labels and dispatch append-only CSV persistence. |
| Health route | `backend/app/api/routes/health.py` | Report model/catalog readiness and live Neo4j or fallback status. |
| Image service | `backend/app/services/image_service.py` | Validate metadata and image bytes, then save a UUID-named JPEG or PNG. |
| Inference adapter | `backend/app/services/inference_service.py` | Connect API schemas to the ML preprocessing and prediction functions. |
| Catalog service | `backend/app/services/catalog_service.py` | Load typed static JSON and retrieve an exact class entry. |
| Feedback service | `backend/app/services/feedback_service.py` | Persist timestamped feedback and recall the latest valid label for an exact image fingerprint. |
| Dataset preparation | `ml/src/data/prepare_dataset.py` | Load the manifest, create strict stratified splits, copy ImageFolder data, and freeze the test manifest. |
| Dataset validation | `ml/src/data/validate_dataset.py` | Verify provenance, image digests, class coverage, exact/near-duplicate split isolation, and the frozen test checksum. |
| Classifier | `ml/src/models/classifier.py` | Build ResNet-18, freeze/unfreeze features, and load validated checkpoints. |
| Training | `ml/src/training/train.py` | Reproducible training, validation, early stopping, best-state restoration, and artifact writing. |
| Evaluation | `ml/src/evaluation/` | Per-class metrics, accuracy, confusion matrix, traceability metadata, and reports. |
| Central configuration | `config/` and `ml/configs/` | Runtime settings, ordered taxonomy, training settings, and evaluation paths. |

## 5. API Contracts

| Method and path | Input | Successful output |
| --- | --- | --- |
| `GET /` | None | `frontend/index.html` |
| `GET /health` | None | Status plus `model_loaded` and `catalog_loaded` flags |
| `POST /api/v1/predict` | Multipart field `file` containing one JPEG or PNG | Prediction identifier, image fingerprint, predicted and closest classes, confidence, correction state, review state, guidance, inventory, and disclaimer |
| `POST /api/v1/feedback` | JSON feedback record | Acceptance status and matching prediction identifier |

The prediction endpoint returns HTTP `503` when no trained model is available,
`415` for unsupported media types, `413` for files above the configured size,
and `400` for invalid metadata or unreadable image bytes.

## 6. Prediction Sequence

1. The UI checks `/health` and enables analysis only when both model and
   catalog are available.
2. The browser validates basic file type and size and sends multipart data.
3. FastAPI enforces the configured media type and 10 MB limit.
4. The image service verifies the filename extension and decodes the bytes with
   Pillow.
5. The service saves the original bytes under a generated UUID filename in
   `storage/uploads`.
6. The backend calculates the image SHA-256 and checks for a saved human label.
7. The ML adapter applies EXIF orientation, converts to RGB, square-pads and
   resizes it, creates a tensor, and applies ImageNet normalization.
8. ResNet-18 emits four logits in the configured class order.
9. Softmax produces a confidence and closest class.
10. Confidence below the checkpoint threshold (`0.65` for v2) becomes `unknown`; the catalog is deliberately not
   consulted for guidance or parts.
11. A saved exact-image label, when present, becomes the disclosed effective
    class without changing the model or presenting the model confidence as
    confidence in the human label.
12. A known class is enriched from Neo4j. If Neo4j is unavailable, the same
    seeded JSON content is returned with `static_catalog_fallback` provenance.
13. The typed response is rendered by the frontend.
14. Optional Y/N feedback is submitted separately and persisted locally.

## 7. Dataset and Artifact Boundaries

- `data/dataset_v2/raw` and `data/dataset_v2/processed` contain the reviewed v2 images.
- `data/dataset_v2/manifests/dataset_manifest.csv` is the accepted source metadata contract.
- Generated split manifests record exact membership.
- `data/dataset_v2/manifests/test_set_checksum.txt` contains the frozen test-manifest
  SHA-256 used by final evaluation.
- `artifacts/v2/models/model.pt` is the promoted versioned checkpoint.
- `artifacts/v2/metrics` and `artifacts/v2/figures` contain generated evidence only;
  they must never contain fabricated results.
- `storage/uploads` is runtime-only local storage and is excluded from Git.

## 8. Reproducibility and Traceability

- The central class order is `normal`, `scratch`, `dent`, `rust`.
- Dataset splitting and training use random seed `42`.
- Training restores the lowest validation-loss model state.
- Checkpoint metadata records architecture, classes, preprocessing, threshold
  provenance, seed, stages, class counts/weights, manifest hashes, and best epoch/loss.
- Final evaluation records the SHA-256 identifiers of both the model file and
  test manifest.

## 9. Security, Privacy, and Failure Behavior

- Uploads are reduced to a safe basename and stored with generated names.
- Declared media type alone is insufficient; Pillow must decode the content.
- Pydantic schemas reject unexpected catalog and feedback fields.
- The application does not execute uploaded content or expose the upload
  directory as a static route.
- Feedback and uploads remain local, but the PoC does not implement encryption,
  user isolation, retention automation, authentication, rate limiting, or
  malware scanning.
- Missing or invalid checkpoints do not produce fabricated predictions; health
  remains degraded and the prediction route is unavailable.

## 10. Deployment Model

The supported local deployment is a Neo4j container plus one application process launched with:

```text
python scripts/run_api.py --port 8000
```

Uvicorn serves the FastAPI application and static frontend on
`http://127.0.0.1:8000` by default. Production hosting, cloud storage,
multi-user concurrency, and external integrations are future concerns and are
not architectural dependencies of this PoC.
