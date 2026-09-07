"""Tests for strict dataset splitting, leakage checks, and freezing."""

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from ml.src.data.prepare_dataset import (
    create_stratified_splits,
    load_dataset_manifest,
    validate_source_images,
    write_split_manifests,
)
from ml.src.data.validate_dataset import (
    detect_duplicates_across_splits,
    detect_near_duplicates_across_splits,
    validate_class_distribution,
    validate_split_ratios,
    validate_traceability,
    verify_manifest_images,
    verify_test_manifest_checksum,
)

CLASS_NAMES = ["normal", "scratch", "dent", "rust"]


def _manifest() -> pd.DataFrame:
    """Create 80 unique records with 20 examples per class."""
    records = []
    for class_name in CLASS_NAMES:
        for index in range(20):
            image_id = f"{class_name}-{index}"
            records.append(
                {
                    "image_id": image_id,
                    "file_path": f"{class_name}/{image_id}.jpg",
                    "class_name": class_name,
                    "checksum": f"checksum-{image_id}",
                }
            )
    return pd.DataFrame(records)


def test_exact_eighty_ten_ten_split() -> None:
    """The fixed fixture should split into exact 80/10/10 counts."""
    result = create_stratified_splits(_manifest(), random_seed=42)
    assert (len(result.train), len(result.validation), len(result.test)) == (64, 8, 8)
    assert result.train["class_name"].value_counts().to_dict() == {
        name: 16 for name in CLASS_NAMES
    }
    assert result.validation["class_name"].value_counts().to_dict() == {
        name: 2 for name in CLASS_NAMES
    }
    assert result.test["class_name"].value_counts().to_dict() == {
        name: 2 for name in CLASS_NAMES
    }


def test_split_is_reproducible() -> None:
    """The same seed should generate identical record membership."""
    first = create_stratified_splits(_manifest(), random_seed=17)
    second = create_stratified_splits(_manifest(), random_seed=17)
    assert first.train["image_id"].tolist() == second.train["image_id"].tolist()
    assert first.validation["image_id"].tolist() == second.validation[
        "image_id"
    ].tolist()
    assert first.test["image_id"].tolist() == second.test["image_id"].tolist()


def test_split_ratio_validation() -> None:
    """Only positive ratios summing to one should be accepted."""
    assert validate_split_ratios(0.8, 0.1, 0.1)
    assert not validate_split_ratios(0.7, 0.1, 0.1)
    assert not validate_split_ratios(0.8, 0.2, 0.0)


def test_strict_split_rejects_incompatible_class_counts() -> None:
    """Strict per-class percentages should reject fractional split counts."""
    incompatible = pd.concat(
        [
            _manifest(),
            pd.DataFrame(
                [
                    {
                        "image_id": "normal-extra",
                        "file_path": "normal/extra.jpg",
                        "class_name": "normal",
                        "checksum": "checksum-normal-extra",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    try:
        create_stratified_splits(incompatible)
    except ValueError as exc:
        assert "compatible with 80/10/10" in str(exc)
        return
    raise AssertionError("Expected strict split to reject a fractional class count")


def test_no_duplicate_image_across_splits(tmp_path: Path) -> None:
    """Independent split manifests should report no leaked identities."""
    result = create_stratified_splits(_manifest())
    write_split_manifests(result, tmp_path)
    duplicates = detect_duplicates_across_splits(
        tmp_path / "train_manifest.csv",
        tmp_path / "validation_manifest.csv",
        tmp_path / "test_manifest.csv",
    )
    assert duplicates == []


def test_duplicate_image_is_detected(tmp_path: Path) -> None:
    """A shared image identity should be reported as leakage."""
    columns = ["image_id", "file_path", "class_name", "checksum"]
    pd.DataFrame(
        [["same", "normal/a.jpg", "normal", "abc"]], columns=columns
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        [["same", "normal/a.jpg", "normal", "abc"]], columns=columns
    ).to_csv(tmp_path / "validation.csv", index=False)
    pd.DataFrame(
        [["other", "normal/b.jpg", "normal", "def"]], columns=columns
    ).to_csv(tmp_path / "test.csv", index=False)
    duplicates = detect_duplicates_across_splits(
        tmp_path / "train.csv",
        tmp_path / "validation.csv",
        tmp_path / "test.csv",
    )
    assert "image_id:same" in duplicates
    assert "checksum:abc" in duplicates


def test_test_manifest_checksum_verification(tmp_path: Path) -> None:
    """The test manifest should be frozen and tampering should be detected."""
    result = create_stratified_splits(_manifest())
    write_split_manifests(result, tmp_path)
    test_manifest = tmp_path / "test_manifest.csv"
    checksum_path = tmp_path / "test_set_checksum.txt"
    assert verify_test_manifest_checksum(test_manifest, checksum_path)
    test_manifest.write_text(
        test_manifest.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )
    assert not verify_test_manifest_checksum(test_manifest, checksum_path)


def test_class_distribution_validation() -> None:
    """All and only the configured classes should be accepted."""
    assert validate_class_distribution(_manifest(), CLASS_NAMES)
    assert not validate_class_distribution(
        _manifest().query("class_name != 'rust'"), CLASS_NAMES
    )


def test_source_manifest_and_image_checksum_validation(tmp_path: Path) -> None:
    """Preparation should admit only present, decodable, checksum-matched images."""
    image_path = tmp_path / "classification" / "normal" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (128, 128), "silver").save(image_path)
    checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "normal-1",
                "file_path": "classification/normal/sample.jpg",
                "class_name": "normal",
                "source": "public-test-source",
                "license": "MIT",
                "checksum": checksum,
                "source_group": "source-image-1.jpg",
                "dataset_split": "",
            }
        ]
    ).to_csv(manifest_path, index=False)

    manifest = load_dataset_manifest(manifest_path)
    validate_source_images(manifest, tmp_path)
    manifest.loc[0, "checksum"] = "0" * 64
    try:
        validate_source_images(manifest, tmp_path)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
        return
    raise AssertionError("Expected a mismatched source checksum to be rejected")


def test_source_group_leakage_is_detected(tmp_path: Path) -> None:
    """Different crops of one original source image must not cross splits."""
    columns = ["image_id", "file_path", "class_name", "checksum", "source_group"]
    pd.DataFrame(
        [["a", "normal/a.jpg", "normal", "a" * 64, "original.jpg"]],
        columns=columns,
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        [["b", "normal/b.jpg", "normal", "b" * 64, "original.jpg"]],
        columns=columns,
    ).to_csv(tmp_path / "validation.csv", index=False)
    pd.DataFrame(
        [["c", "normal/c.jpg", "normal", "c" * 64, "other.jpg"]],
        columns=columns,
    ).to_csv(tmp_path / "test.csv", index=False)
    duplicates = detect_duplicates_across_splits(
        tmp_path / "train.csv",
        tmp_path / "validation.csv",
        tmp_path / "test.csv",
    )
    assert "source_group:original.jpg" in duplicates


def test_cross_split_perceptual_duplicate_is_detected(tmp_path: Path) -> None:
    """Near-identical perceptual hashes across splits must stop preparation."""
    columns = ["image_id", "class_name", "perceptual_hash"]
    pd.DataFrame([["a", "normal", "0000000000000000"]], columns=columns).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame([["b", "normal", "0000000000000001"]], columns=columns).to_csv(
        tmp_path / "validation.csv", index=False
    )
    pd.DataFrame([["c", "normal", "ffffffffffffffff"]], columns=columns).to_csv(
        tmp_path / "test.csv", index=False
    )
    matches = detect_near_duplicates_across_splits(
        tmp_path / "train.csv",
        tmp_path / "validation.csv",
        tmp_path / "test.csv",
        maximum_hamming_distance=1,
    )
    assert len(matches) == 1
    assert "train:a<->validation:b" in matches[0]


def test_traceability_and_image_digest_verification(tmp_path: Path) -> None:
    """Version-two records must bind provenance and both image digests."""
    from ml.src.data.image_integrity import file_difference_hash

    image_path = tmp_path / "normal" / "sample.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (128, 96), "navy").save(image_path)
    checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = pd.DataFrame(
        [
            {
                "image_id": "normal-1",
                "file_path": "normal/sample.jpg",
                "class_name": "normal",
                "original_label": "undamaged car",
                "source_dataset": "test source",
                "source_url": "https://example.test/image",
                "licence": "CC0-1.0",
                "checksum": checksum,
                "perceptual_hash": file_difference_hash(image_path),
                "dataset_split": "train",
            }
        ]
    )
    assert validate_traceability(manifest) == []
    assert verify_manifest_images(manifest, tmp_path) == []
    manifest.loc[0, "checksum"] = "0" * 64
    assert "SHA-256 mismatch" in verify_manifest_images(manifest, tmp_path)[0]
