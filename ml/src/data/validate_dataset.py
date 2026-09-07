"""Dataset integrity checks for the classification pipeline."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from ml.src.data.image_integrity import (
    file_difference_hash,
    file_sha256,
    hamming_distance,
)

TRACEABILITY_COLUMNS = {
    "image_id",
    "file_path",
    "class_name",
    "original_label",
    "source_dataset",
    "source_url",
    "licence",
    "checksum",
    "perceptual_hash",
    "dataset_split",
}


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> bool:
    """Validate positive train/validation/test ratios that sum to one."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not isinstance(value, (int, float)) for value in ratios):
        return False
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        return False
    return math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9)


def _read_manifest(manifest_path: Path) -> DataFrame:
    """Read a non-empty CSV manifest or raise a descriptive error."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    # String dtype preserves leading zeroes in hexadecimal perceptual hashes.
    manifest = pd.read_csv(manifest_path, dtype=str)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return manifest


def detect_duplicates_across_splits(
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
) -> list[str]:
    """Return identities that occur in more than one dataset split.

    Images are compared by any available non-empty ``checksum``, ``image_id``,
    and normalized ``file_path`` values. Returning every matching identity makes
    accidental leakage easier to diagnose than returning only a Boolean.
    """
    manifests = {
        "train": _read_manifest(train_manifest),
        "validation": _read_manifest(validation_manifest),
        "test": _read_manifest(test_manifest),
    }
    locations: dict[str, set[str]] = {}

    for split_name, manifest in manifests.items():
        for column in ("checksum", "image_id", "file_path", "source_group"):
            if column not in manifest.columns:
                continue
            values = manifest[column].dropna().astype(str).str.strip()
            for value in values[values != ""]:
                normalized = value.casefold() if column == "file_path" else value
                identity = f"{column}:{normalized}"
                locations.setdefault(identity, set()).add(split_name)

    return sorted(
        identity for identity, split_names in locations.items() if len(split_names) > 1
    )


def detect_near_duplicates_across_splits(
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    maximum_hamming_distance: int = 6,
) -> list[str]:
    """Return perceptually similar records that cross split boundaries.

    A 64-bit difference hash is deliberately treated as a review signal. The
    threshold is configurable because crops with uniform paint can otherwise
    look deceptively similar to a compact perceptual hash.
    """
    if not 0 <= maximum_hamming_distance <= 64:
        raise ValueError("maximum_hamming_distance must be between 0 and 64")
    records: list[tuple[str, str, str]] = []
    for split_name, manifest_path in (
        ("train", train_manifest),
        ("validation", validation_manifest),
        ("test", test_manifest),
    ):
        manifest = _read_manifest(manifest_path)
        if "perceptual_hash" not in manifest.columns:
            continue
        for _, row in manifest.iterrows():
            perceptual_hash = str(row.get("perceptual_hash", "")).strip().casefold()
            if not perceptual_hash:
                continue
            if len(perceptual_hash) != 16 or any(
                character not in "0123456789abcdef" for character in perceptual_hash
            ):
                raise ValueError(
                    f"Invalid perceptual_hash in {manifest_path}: {perceptual_hash}"
                )
            records.append(
                (split_name, str(row.get("image_id", "unknown")), perceptual_hash)
            )

    matches: list[str] = []
    for index, (first_split, first_id, first_hash) in enumerate(records):
        for second_split, second_id, second_hash in records[index + 1 :]:
            if first_split == second_split:
                continue
            distance = hamming_distance(first_hash, second_hash)
            if distance <= maximum_hamming_distance:
                matches.append(
                    f"{first_split}:{first_id}<->{second_split}:{second_id} "
                    f"(dHash distance {distance})"
                )
    return sorted(matches)


def validate_traceability(manifest: DataFrame) -> list[str]:
    """Return traceability violations for a version-two dataset manifest."""
    errors: list[str] = []
    missing = TRACEABILITY_COLUMNS - set(manifest.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    for column in TRACEABILITY_COLUMNS:
        values = manifest[column].fillna("").astype(str).str.strip()
        if (values == "").any():
            errors.append(f"blank values in {column}")
    if not manifest["checksum"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        errors.append("checksum values must be 64 hexadecimal characters")
    if not manifest["perceptual_hash"].astype(str).str.fullmatch(
        r"[0-9a-fA-F]{16}"
    ).all():
        errors.append("perceptual_hash values must be 16 hexadecimal characters")
    return errors


def verify_manifest_images(manifest: DataFrame, images_root: Path) -> list[str]:
    """Return checksum, perceptual-hash, and decoding errors for image rows."""
    errors: list[str] = []
    root = images_root.resolve()
    for _, row in manifest.iterrows():
        relative = Path(str(row.get("file_path", "")))
        image_path = relative if relative.is_absolute() else root / relative
        image_id = str(row.get("image_id", image_path.name))
        if not image_path.is_file():
            errors.append(f"{image_id}: image is missing at {image_path}")
            continue
        try:
            actual_checksum = file_sha256(image_path)
            actual_perceptual_hash = file_difference_hash(image_path)
        except (OSError, ValueError) as exc:
            errors.append(f"{image_id}: image could not be decoded ({exc})")
            continue
        expected_checksum = str(row.get("checksum", "")).strip().casefold()
        if expected_checksum and actual_checksum != expected_checksum:
            errors.append(f"{image_id}: SHA-256 mismatch")
        expected_perceptual_hash = str(row.get("perceptual_hash", "")).strip().casefold()
        if expected_perceptual_hash and actual_perceptual_hash != expected_perceptual_hash:
            errors.append(f"{image_id}: perceptual hash mismatch")
    return errors


def validate_class_distribution(
    manifest: DataFrame,
    expected_classes: list[str],
) -> bool:
    """Check that a manifest contains every expected class and no unknown class."""
    if "class_name" not in manifest.columns or manifest.empty:
        return False
    expected = {name.strip() for name in expected_classes if name.strip()}
    observed = set(manifest["class_name"].dropna().astype(str).str.strip())
    return bool(expected) and observed == expected


def calculate_manifest_checksum(manifest_path: Path) -> str:
    """Return the SHA-256 checksum of a manifest's exact bytes."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    digest = hashlib.sha256()
    with manifest_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_test_manifest_checksum(
    manifest_path: Path,
    expected_checksum_path: Path,
) -> bool:
    """Verify that the frozen test manifest still matches its saved checksum."""
    if not expected_checksum_path.is_file():
        raise FileNotFoundError(
            f"Checksum file does not exist: {expected_checksum_path}"
        )
    tokens = expected_checksum_path.read_text(encoding="utf-8").strip().split()
    if not tokens:
        return False
    expected = tokens[0]
    actual = calculate_manifest_checksum(manifest_path)
    return hmac.compare_digest(actual, expected)


def main() -> None:
    """Validate three split manifests from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-checksum", type=Path, required=True)
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("config/class_names.json"),
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        help="Optional root used to recheck every manifest image and digest.",
    )
    parser.add_argument(
        "--require-traceability",
        action="store_true",
        help="Require the complete version-two provenance schema.",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=6,
        help="Maximum 64-bit dHash distance treated as cross-split leakage.",
    )
    args = parser.parse_args()

    duplicates = detect_duplicates_across_splits(
        args.train_manifest,
        args.validation_manifest,
        args.test_manifest,
    )
    if duplicates:
        raise SystemExit(f"Cross-split duplicates found: {duplicates}")
    near_duplicates = detect_near_duplicates_across_splits(
        args.train_manifest,
        args.validation_manifest,
        args.test_manifest,
        args.near_duplicate_distance,
    )
    if near_duplicates:
        raise SystemExit(f"Cross-split near duplicates found: {near_duplicates}")
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not class_names:
        raise SystemExit("Class-name configuration must be a non-empty JSON list")
    manifests = (
        ("train", _read_manifest(args.train_manifest)),
        ("validation", _read_manifest(args.validation_manifest)),
        ("test", _read_manifest(args.test_manifest)),
    )
    expected_classes = [str(value) for value in class_names]
    for split_name, manifest in manifests:
        if args.require_traceability:
            traceability_errors = validate_traceability(manifest)
            if traceability_errors:
                raise SystemExit(
                    f"{split_name} manifest traceability failed: "
                    f"{traceability_errors}"
                )
        if args.images_root is not None:
            image_errors = verify_manifest_images(manifest, args.images_root)
            if image_errors:
                raise SystemExit(
                    f"{split_name} manifest image verification failed: "
                    f"{image_errors}"
                )
        if not validate_class_distribution(manifest, expected_classes):
            raise SystemExit(
                f"{split_name} manifest does not contain exactly the configured classes"
            )
        if "dataset_split" not in manifest.columns or set(
            manifest["dataset_split"].astype(str)
        ) != {split_name}:
            raise SystemExit(
                f"{split_name} manifest contains incorrect dataset_split values"
            )

    combined = pd.concat(
        [manifest.assign(_split=split_name) for split_name, manifest in manifests],
        ignore_index=True,
    )
    for class_name in expected_classes:
        class_rows = combined[combined["class_name"] == class_name]
        counts = class_rows["_split"].value_counts().to_dict()
        total = len(class_rows)
        expected_counts = {
            "train": int(total * 0.8),
            "validation": int(total * 0.1),
            "test": int(total * 0.1),
        }
        if counts != expected_counts:
            raise SystemExit(
                f"Strict split counts are incorrect for {class_name}: {counts}"
            )
    if not verify_test_manifest_checksum(args.test_manifest, args.test_checksum):
        raise SystemExit("Frozen test manifest checksum does not match")
    print("Dataset split integrity checks passed.")


if __name__ == "__main__":
    main()
