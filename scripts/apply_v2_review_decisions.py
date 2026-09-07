"""Apply the recorded pre-inference visual-review decisions to source records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CLASS_ORDER = ["normal", "scratch", "dent", "rust"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=Path("data/dataset_v2/source_records.csv")
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/dataset_v2/review_decisions.json"),
    )
    args = parser.parse_args()
    records = pd.read_csv(args.records, keep_default_na=False)
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    accepted = decisions["accepted_class_indices"]
    if set(accepted) != set(CLASS_ORDER):
        raise ValueError("Review decisions must cover the ordered four classes")
    records["review_status"] = "rejected"
    protocol = str(decisions["review_protocol"])
    for class_name in CLASS_ORDER:
        row_indices = list(records.index[records["class_name"] == class_name])
        accepted_indices = [int(value) for value in accepted[class_name]]
        if len(accepted_indices) != len(set(accepted_indices)):
            raise ValueError(f"Duplicate accepted index for {class_name}")
        if not accepted_indices or min(accepted_indices) < 0 or max(accepted_indices) >= len(row_indices):
            raise ValueError(f"Accepted index is out of range for {class_name}")
        accepted_rows = [row_indices[index] for index in accepted_indices]
        records.loc[accepted_rows, "review_status"] = "accepted"
        records.loc[accepted_rows, "review_notes"] = (
            records.loc[accepted_rows, "review_notes"].astype(str)
            + f"; accepted under {protocol}"
        )
        rejected_rows = [index for index in row_indices if index not in accepted_rows]
        records.loc[rejected_rows, "review_notes"] = (
            records.loc[rejected_rows, "review_notes"].astype(str)
            + f"; rejected under {protocol}: failed one or more visual inclusion rules"
        )
    records.to_csv(args.records, index=False)
    print(records.groupby(["class_name", "review_status"]).size().to_string())


if __name__ == "__main__":
    main()
