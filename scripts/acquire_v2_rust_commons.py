"""Append pre-inference Wikimedia corrosion candidates to dataset v2."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.acquire_dataset_v2_candidates import (  # noqa: E402
    CLASS_ORDER,
    RECORD_COLUMNS,
    _commons_candidates,
    _save_commons_candidate,
)

CATEGORY = "Rusty automobiles"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--records", type=Path, default=Path("data/dataset_v2/source_records.csv")
    )
    parser.add_argument(
        "--raw-directory", type=Path, default=Path("data/dataset_v2/raw")
    )
    args = parser.parse_args()
    existing = pd.read_csv(args.records, keep_default_na=False)
    if set(existing.columns) != set(RECORD_COLUMNS):
        raise ValueError("Existing candidate manifest has an unexpected schema")
    excluded_groups = set(existing["source_group"].astype(str))
    demo = pd.read_csv(PROJECT_ROOT / "demo_images" / "demo_manifest.csv")
    excluded_groups.update(demo["source_group"].astype(str))
    candidates = _commons_candidates(args.count, excluded_groups, CATEGORY)
    with ThreadPoolExecutor(max_workers=8) as executor:
        new_rows = [
            row
            for row in executor.map(
                lambda record: _save_commons_candidate(
                    record, args.raw_directory, "rust", CATEGORY
                ),
                candidates,
            )
            if row is not None
        ]
    if len(new_rows) < max(10, int(args.count * 0.8)):
        raise RuntimeError(
            f"Only {len(new_rows)} of {args.count} candidates decoded successfully; "
            "too few remain for meaningful review"
        )
    rows = existing.to_dict("records") + new_rows
    rows.sort(key=lambda row: (CLASS_ORDER.index(str(row["class_name"])), row["file_path"]))
    with args.records.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Appended {len(new_rows)} {CATEGORY} candidates with review_status=pending.")


if __name__ == "__main__":
    main()
