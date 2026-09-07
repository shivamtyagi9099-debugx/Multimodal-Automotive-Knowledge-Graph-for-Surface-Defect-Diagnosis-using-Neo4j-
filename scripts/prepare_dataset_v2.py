"""Prepare a human-reviewed, traceable version-two dataset and frozen split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.data.image_integrity import (  # noqa: E402
    file_difference_hash,
    file_sha256,
    hamming_distance,
)
from ml.src.data.prepare_dataset import (  # noqa: E402
    copy_split_files,
    create_stratified_splits,
    load_dataset_manifest,
    validate_source_images,
    write_split_manifests,
)

INPUT_COLUMNS = {
    "file_path",
    "class_name",
    "original_label",
    "source_dataset",
    "source_url",
    "licence",
    "source_group",
    "review_status",
    "review_notes",
}
CLASS_ORDER = ["normal", "scratch", "dent", "rust"]


def build_reviewed_manifest(
    records_path: Path,
    raw_directory: Path,
    near_duplicate_distance: int,
) -> pd.DataFrame:
    """Bind accepted human-review records to measured image identities."""
    records = pd.read_csv(records_path, keep_default_na=False)
    missing = INPUT_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(f"Source records are missing columns: {sorted(missing)}")
    accepted = records[
        records["review_status"].astype(str).str.strip().str.casefold() == "accepted"
    ].copy()
    if accepted.empty:
        raise ValueError("No records have review_status=accepted")
    if not set(accepted["class_name"]).issubset(CLASS_ORDER):
        raise ValueError("Accepted rows contain an unsupported class")
    if set(accepted["class_name"]) != set(CLASS_ORDER):
        raise ValueError("Accepted rows must cover all four ordered classes")
    for column in INPUT_COLUMNS - {"review_notes"}:
        if (accepted[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Accepted rows contain blank {column} values")
    if accepted["source_group"].duplicated().any():
        raise ValueError(
            "Only one accepted image is allowed per source_group; retain a "
            "single representative from related sequences"
        )

    output_rows: list[dict[str, str]] = []
    for _, record in accepted.sort_values(
        ["class_name", "source_dataset", "source_group", "file_path"]
    ).iterrows():
        relative_path = Path(str(record["file_path"]))
        path = relative_path if relative_path.is_absolute() else raw_directory / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Reviewed image is missing: {path}")
        checksum = file_sha256(path)
        perceptual_hash = file_difference_hash(path)
        class_name = str(record["class_name"])
        output_rows.append(
            {
                "image_id": f"{class_name}-{checksum[:16]}",
                "file_path": relative_path.as_posix(),
                "class_name": class_name,
                "original_label": str(record["original_label"]),
                "source_dataset": str(record["source_dataset"]),
                "source_url": str(record["source_url"]),
                "licence": str(record["licence"]),
                "checksum": checksum,
                "perceptual_hash": perceptual_hash,
                "dataset_split": "",
                "source_group": str(record["source_group"]),
                "review_status": "accepted",
                "review_notes": str(record["review_notes"]),
                # Legacy aliases keep the existing preparation library usable.
                "source": str(record["source_dataset"]),
                "license": str(record["licence"]),
            }
        )
    manifest = pd.DataFrame(output_rows)
    if manifest["checksum"].duplicated().any():
        duplicates = manifest.loc[
            manifest["checksum"].duplicated(keep=False),
            ["image_id", "checksum"],
        ].to_dict("records")
        raise ValueError(f"Exact duplicate images remain: {duplicates}")

    rows = manifest.to_dict("records")
    near_duplicates: list[dict[str, object]] = []
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            distance = hamming_distance(
                str(first["perceptual_hash"]), str(second["perceptual_hash"])
            )
            if distance <= near_duplicate_distance:
                near_duplicates.append(
                    {
                        "first_image_id": first["image_id"],
                        "second_image_id": second["image_id"],
                        "first_class": first["class_name"],
                        "second_class": second["class_name"],
                        "hamming_distance": distance,
                    }
                )
    if near_duplicates:
        raise ValueError(
            "Perceptual near-duplicate candidates remain; review and reject one "
            f"of each pair before splitting: {json.dumps(near_duplicates[:20])}"
        )

    counts = manifest["class_name"].value_counts().to_dict()
    incompatible = {
        class_name: int(count)
        for class_name, count in counts.items()
        if count < 10 or count % 10 != 0
    }
    if incompatible:
        raise ValueError(
            "Accepted per-class counts must be at least 10 and multiples of 10 "
            f"for the strict 80/10/10 split: {incompatible}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/dataset_v2/source_records.csv"),
    )
    parser.add_argument(
        "--raw-directory", type=Path, default=Path("data/dataset_v2/raw")
    )
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=Path("data/dataset_v2/processed"),
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=Path("data/dataset_v2/manifests"),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=6)
    args = parser.parse_args()

    manifest = build_reviewed_manifest(
        args.records, args.raw_directory, args.near_duplicate_distance
    )
    args.manifest_directory.mkdir(parents=True, exist_ok=True)
    source_manifest = args.manifest_directory / "dataset_manifest.csv"
    manifest.to_csv(source_manifest, index=False)
    validated = load_dataset_manifest(source_manifest)
    validate_source_images(validated, args.raw_directory)
    splits = create_stratified_splits(validated, random_seed=args.random_seed)
    copy_split_files(splits, args.raw_directory, args.processed_directory)
    write_split_manifests(splits, args.manifest_directory)
    print("Dataset v2 prepared from human-reviewed records.")
    print(json.dumps(manifest["class_name"].value_counts().to_dict(), indent=2))


if __name__ == "__main__":
    main()
