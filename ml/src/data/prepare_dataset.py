"""Manifest-driven, reproducible dataset preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from ml.src.data.validate_dataset import (
    calculate_manifest_checksum,
    validate_split_ratios,
)

REQUIRED_COLUMNS = {
    "image_id",
    "file_path",
    "class_name",
    "source",
    "license",
    "checksum",
}


@dataclass(frozen=True)
class SplitResult:
    """Contain reproducible train, validation, and untouched test records."""

    train: DataFrame
    validation: DataFrame
    test: DataFrame


def load_dataset_manifest(manifest_path: Path) -> DataFrame:
    """Load and validate the source dataset manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing_columns = REQUIRED_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    if manifest.empty:
        raise ValueError("Dataset manifest contains no image records")

    manifest = manifest.copy()
    for column in REQUIRED_COLUMNS:
        manifest[column] = manifest[column].fillna("").astype(str).str.strip()
    empty_columns = sorted(
        column for column in REQUIRED_COLUMNS if (manifest[column] == "").any()
    )
    if empty_columns:
        raise ValueError(f"Required manifest values are blank: {empty_columns}")
    invalid_checksums = manifest.loc[
        ~manifest["checksum"].str.fullmatch(r"[0-9a-fA-F]{64}"), "checksum"
    ].tolist()
    if invalid_checksums:
        raise ValueError("Manifest checksums must be 64-character SHA-256 values")
    if manifest["file_path"].duplicated().any():
        duplicates = manifest.loc[
            manifest["file_path"].duplicated(keep=False), "file_path"
        ].tolist()
        raise ValueError(f"Manifest contains duplicate file paths: {duplicates}")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image_id values")
    if manifest["checksum"].duplicated().any():
        raise ValueError("Manifest contains duplicate image checksums")
    if "source_group" in manifest.columns:
        source_groups = manifest["source_group"].fillna("").astype(str).str.strip()
        non_empty_groups = source_groups[source_groups != ""]
        if non_empty_groups.duplicated().any():
            raise ValueError(
                "Manifest contains multiple crops from the same source_group"
            )
    if "dataset_split" in manifest.columns:
        assigned = manifest["dataset_split"].fillna("").astype(str).str.strip()
        if (assigned != "").any():
            raise ValueError(
                "Source manifest dataset_split values must be blank before splitting"
            )
    return manifest


def validate_source_images(manifest: DataFrame, source_directory: Path) -> None:
    """Verify every source-relative image, checksum, and decodability."""
    source_directory = source_directory.resolve()
    for _, record in manifest.iterrows():
        recorded_path = Path(str(record["file_path"]))
        image_path = (
            recorded_path
            if recorded_path.is_absolute()
            else source_directory / recorded_path
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image does not exist: {image_path}")
        actual_checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_checksum.casefold() != str(record["checksum"]).casefold():
            raise ValueError(f"Source image checksum mismatch: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Source image cannot be decoded: {image_path}") from exc


def create_stratified_splits(
    manifest: DataFrame,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> SplitResult:
    """Create reproducible, class-stratified train/validation/test splits."""
    if not validate_split_ratios(train_ratio, validation_ratio, test_ratio):
        raise ValueError("Split ratios must be positive and sum exactly to 1.0")
    if "class_name" not in manifest.columns or manifest.empty:
        raise ValueError("A non-empty class_name column is required")

    class_counts = manifest["class_name"].value_counts()
    if (class_counts < 10).any():
        too_small = class_counts[class_counts < 10].to_dict()
        raise ValueError(
            "Each class needs at least 10 images for a strict 80/10/10 split; "
            f"insufficient classes: {too_small}"
        )
    ratios = (train_ratio, validation_ratio, test_ratio)
    non_divisible = {
        class_name: int(count)
        for class_name, count in class_counts.items()
        if any(
            not math.isclose(count * ratio, round(count * ratio), abs_tol=1e-9)
            for ratio in ratios
        )
    }
    if non_divisible:
        raise ValueError(
            "Strict per-class split counts require each class count to be "
            "compatible with 80/10/10 (normally a multiple of 10); "
            f"incompatible classes: {non_divisible}"
        )

    train, remainder = train_test_split(
        manifest,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=manifest["class_name"],
    )
    relative_validation_ratio = validation_ratio / (validation_ratio + test_ratio)
    validation, test = train_test_split(
        remainder,
        train_size=relative_validation_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=remainder["class_name"],
    )

    split_frames: dict[str, DataFrame] = {}
    for split_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        prepared = frame.copy().sort_index().reset_index(drop=True)
        prepared["dataset_split"] = split_name
        split_frames[split_name] = prepared

    return SplitResult(
        train=split_frames["train"],
        validation=split_frames["validation"],
        test=split_frames["test"],
    )


def _safe_destination_name(record: pd.Series, source_path: Path) -> str:
    """Build a deterministic destination filename without unsafe characters."""
    image_id = str(record.get("image_id", "")).strip()
    if not image_id:
        return source_path.name
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_id)
    return f"{safe_id}__{source_path.name}"


def copy_split_files(
    split_result: SplitResult,
    source_directory: Path,
    destination_directory: Path,
) -> None:
    """Copy split images into ``split/class_name`` ImageFolder directories."""
    source_directory = source_directory.resolve()
    destination_directory.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        path
        for path in destination_directory.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    if existing_outputs:
        raise FileExistsError(
            "Processed dataset already contains generated files; use an empty "
            f"destination directory: {destination_directory}"
        )

    for split_name, manifest in (
        ("train", split_result.train),
        ("validation", split_result.validation),
        ("test", split_result.test),
    ):
        for _, record in manifest.iterrows():
            recorded_path = Path(str(record["file_path"]))
            source_path = (
                recorded_path
                if recorded_path.is_absolute()
                else source_directory / recorded_path
            )
            if not source_path.is_file():
                raise FileNotFoundError(f"Source image does not exist: {source_path}")

            class_name = str(record["class_name"])
            class_directory = destination_directory / split_name / class_name
            class_directory.mkdir(parents=True, exist_ok=True)
            destination_path = class_directory / _safe_destination_name(
                record, source_path
            )
            shutil.copy2(source_path, destination_path)


def write_split_manifests(
    split_result: SplitResult,
    output_directory: Path,
) -> None:
    """Write split manifests and freeze the test manifest with SHA-256."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_directory / "train_manifest.csv",
        "validation": output_directory / "validation_manifest.csv",
        "test": output_directory / "test_manifest.csv",
    }
    split_result.train.to_csv(paths["train"], index=False)
    split_result.validation.to_csv(paths["validation"], index=False)
    split_result.test.to_csv(paths["test"], index=False)

    checksum = calculate_manifest_checksum(paths["test"])
    (output_directory / "test_set_checksum.txt").write_text(
        f"{checksum}\n", encoding="utf-8"
    )


def main() -> None:
    """Create processed split directories and frozen manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--processed-directory", type=Path, required=True)
    parser.add_argument("--manifest-output-directory", type=Path, required=True)
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("config/class_names.json"),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest)
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Class-name configuration must be a non-empty JSON list")
    configured_classes = {str(value).strip() for value in class_names}
    observed_classes = set(manifest["class_name"])
    if observed_classes != configured_classes:
        raise ValueError(
            "Source manifest classes do not match config/class_names.json: "
            f"observed={sorted(observed_classes)}"
        )
    validate_source_images(manifest, args.source_directory)
    splits = create_stratified_splits(manifest, random_seed=args.random_seed)
    copy_split_files(splits, args.source_directory, args.processed_directory)
    write_split_manifests(splits, args.manifest_output_directory)
    print(
        "Prepared dataset: "
        f"train={len(splits.train)}, "
        f"validation={len(splits.validation)}, test={len(splits.test)}"
    )


if __name__ == "__main__":
    main()
