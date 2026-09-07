"""Validate frozen demo provenance and evaluate every image through the API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.data.image_integrity import (
    file_difference_hash,
    file_sha256,
    hamming_distance,
)

CLASS_ORDER = ["normal", "scratch", "dent", "rust"]
REQUIRED_COLUMNS = {
    "filename",
    "expected_class",
    "source_dataset",
    "source_url",
    "licence",
    "checksum",
    "perceptual_hash",
    "excluded_from_training_validation",
    "selected_before_inference",
    "selection_protocol_version",
}


def _verify_manifest(
    manifest: pd.DataFrame,
    directory: Path,
    development_manifest_path: Path | None,
    near_duplicate_distance: int,
) -> None:
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Demo manifest is missing columns: {sorted(missing)}")
    if manifest.empty or set(manifest["expected_class"]) != set(CLASS_ORDER):
        raise ValueError("Demo manifest must contain all four exact classes")
    counts = manifest["expected_class"].value_counts().to_dict()
    if counts != {class_name: 10 for class_name in CLASS_ORDER}:
        raise ValueError(f"Demo manifest must contain 10 images per class: {counts}")
    if not (
        manifest["excluded_from_training_validation"].astype(str).str.casefold()
        == "yes"
    ).all():
        raise ValueError("Every demo row must be excluded from training/validation")
    if not (
        manifest["selected_before_inference"].astype(str).str.casefold() == "yes"
    ).all():
        raise ValueError("Demo selection must be frozen before inference")
    for _, row in manifest.iterrows():
        image_path = directory / str(row["filename"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Demo image is missing: {image_path}")
        if file_sha256(image_path) != str(row["checksum"]).casefold():
            raise ValueError(f"Demo SHA-256 mismatch: {image_path}")
        if file_difference_hash(image_path) != str(row["perceptual_hash"]).casefold():
            raise ValueError(f"Demo perceptual hash mismatch: {image_path}")
    if development_manifest_path is None:
        return
    development = pd.read_csv(development_manifest_path, dtype=str).fillna("")
    if "checksum" not in development or "perceptual_hash" not in development:
        raise ValueError("Development manifest lacks checksum or perceptual_hash")
    if "source_group" in manifest and "source_group" in development:
        group_overlap = set(manifest["source_group"].astype(str)) & set(
            development["source_group"].astype(str)
        )
        if group_overlap:
            raise ValueError(
                f"Demo/development source groups overlap: {sorted(group_overlap)}"
            )
    checksum_overlap = set(manifest["checksum"].astype(str)) & set(
        development["checksum"].astype(str)
    )
    if checksum_overlap:
        raise ValueError("Demo and development data contain exact image duplicates")
    for _, demo_row in manifest.iterrows():
        for _, development_row in development.iterrows():
            distance = hamming_distance(
                str(demo_row["perceptual_hash"]),
                str(development_row["perceptual_hash"]),
            )
            if distance <= near_duplicate_distance:
                raise ValueError(
                    "Demo/development perceptual near duplicate: "
                    f"{demo_row['filename']} and "
                    f"{development_row.get('image_id', 'unknown')} "
                    f"(dHash distance {distance})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("demo_images"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output", type=Path, default=Path("demo_images/demo_results.csv")
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/dataset_v2/manifests/dataset_manifest.csv"),
    )
    parser.add_argument("--near-duplicate-distance", type=int, default=6)
    args = parser.parse_args()
    manifest = pd.read_csv(args.directory / "demo_manifest.csv", keep_default_na=False)
    _verify_manifest(
        manifest,
        args.directory,
        args.development_manifest,
        args.near_duplicate_distance,
    )

    health = httpx.get(f"{args.api_url.rstrip('/')}/health", timeout=15.0)
    health.raise_for_status()
    if not health.json().get("model_loaded"):
        raise SystemExit("API health reports that the model is not loaded")

    results: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        image_path = args.directory / str(row["filename"])
        media_type = "image/png" if image_path.suffix.casefold() == ".png" else "image/jpeg"
        with image_path.open("rb") as stream:
            response = httpx.post(
                f"{args.api_url.rstrip('/')}/api/v1/predict",
                files={"file": (image_path.name, stream, media_type)},
                timeout=60.0,
            )
        response.raise_for_status()
        prediction = response.json()
        if prediction.get("feedback_applied"):
            raise RuntimeError(
                f"Saved feedback altered demo prediction for {image_path.name}"
            )
        predicted_class = str(prediction["predicted_class"])
        passed = predicted_class == str(row["expected_class"])
        result = {
            "filename": str(row["filename"]),
            "expected_class": str(row["expected_class"]),
            "predicted_class": predicted_class,
            "most_likely_class": str(prediction["most_likely_class"]),
            "confidence": float(prediction["confidence"]),
            "requires_manual_review": bool(prediction["requires_manual_review"]),
            "pass": passed,
        }
        results.append(result)
        print(
            f"{result['filename']}: expected={result['expected_class']} "
            f"predicted={predicted_class} confidence={result['confidence']:.4f} "
            f"{'PASS' if passed else 'FAIL'}"
        )
    result_frame = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.output, index=False)
    overall = float(result_frame["pass"].mean())
    per_class = {
        class_name: float(
            result_frame.loc[
                result_frame["expected_class"] == class_name, "pass"
            ].mean()
        )
        for class_name in CLASS_ORDER
    }
    summary = {
        "api_url": args.api_url,
        "sample_count": len(result_frame),
        "overall_accuracy": overall,
        "per_class_accuracy": per_class,
    }
    summary_path = args.output.with_name("demo_results_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
