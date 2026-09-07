# Dataset v2 admission gate and frozen status

This directory is intentionally separate from the original 120-image baseline.
The current version is complete and frozen: 497 candidates were reviewed,
147 were rejected, and 350 were accepted (normal 80, scratch 100, dent 100,
rust 70). The split is train 280, validation 35, and test 35. Every accepted
row has a pinned source, licence, SHA-256, dHash, review decision, and split.

For a future dataset version, images may enter `raw/<class>/` only after their
public source and licence have been verified and a human has confirmed that the
image meets the visual rules in `docs/dataset_card.md`.

1. Put candidate files under `raw/normal`, `raw/scratch`, `raw/dent`, or
   `raw/rust`.
2. Add one row per candidate to `source_records.csv`. Use one shared
   `source_group` for related frames and accept only one representative from
   that group.
3. Record `accepted` or `rejected` in `review_status` before any inference.
4. Freeze the decisions in `review_decisions.json`, apply them with
   `python scripts/apply_v2_review_decisions.py`, then run
   `python scripts/prepare_dataset_v2.py`.

The command measures SHA-256 and perceptual dHash values, rejects exact and
near-duplicate candidates, creates the strict 80/10/10 split, and freezes the
test manifest. It never consults model predictions. Acquisition refuses to
overwrite the current pool; create `dataset_v3` rather than silently changing
this version.
