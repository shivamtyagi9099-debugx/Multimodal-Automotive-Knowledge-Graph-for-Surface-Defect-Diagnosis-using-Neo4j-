# Frozen presentation-demo images

These 40 images were selected on 2026-09-06 using visual rules before running
model inference: one clear car exterior, one unambiguous expected class,
adequate resolution, no diagram or watermark, and no dominant competing defect.
The selection is deliberately independent of model output.

- `normal` and `rust` are individually licensed Wikimedia Commons files; full
  source pages, creators, and licence URLs are recorded in `demo_manifest.csv`.
- `scratch` and `dent` are held-out crops from the Humans in the Loop Car Parts
  and Car Damages Dataset, published under CC0 1.0 and mirrored at an immutable
  Hugging Face revision.
- Every row records SHA-256, perceptual dHash, source group, and confirmation
  that it was excluded from training and validation.
- `scripts/build_demo_images.py` encodes the pre-inference selection and refuses
  to overwrite an existing frozen folder.
- `scripts/validate_demo_images.py` verifies the manifest and sends every image
  through the running API. It also rejects exact or near-duplicate overlap with
  dataset v2. Failed images remain in place and are reported.

Run the API first, then execute:

```text
python scripts/validate_demo_images.py --api-url http://127.0.0.1:8000
```

The frozen result is 28/40 (70.00%) for v2: normal 9/10, scratch 4/10,
dent 5/10, and rust 10/10. The preserved baseline scores 15/40 (37.50%) on
these exact images. This 32.5 percentage-point gain is the valid direct
before/after comparison. See `artifacts/v2/metrics/demo_results.csv` for all
40 predictions, including every failure.

The set is for an academic demonstration only. Wikimedia reuse must retain the
attribution and licence details in the manifest.
