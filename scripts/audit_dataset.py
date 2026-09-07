"""Generate reproducible evidence about dataset quality and leakage risks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.data.image_integrity import (  # noqa: E402
    file_difference_hash,
    hamming_distance,
    normalized_rgb,
)


def _load_split_manifests(manifest_directory: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name in ("train", "validation", "test"):
        path = manifest_directory / f"{split_name}_manifest.csv"
        frame = pd.read_csv(path)
        frame["audit_split"] = split_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def audit_dataset(
    manifest_directory: Path,
    images_root: Path,
    near_duplicate_distance: int = 6,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return summary evidence and all perceptual near-duplicate candidates."""
    manifest = _load_split_manifests(manifest_directory)
    records: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        recorded_path = Path(str(row["file_path"]))
        image_path = recorded_path if recorded_path.is_absolute() else images_root / recorded_path
        with Image.open(image_path) as source:
            orientation = int(source.getexif().get(274, 1))
            image = normalized_rgb(source)
            luma_extrema = ImageStat.Stat(image.convert("L").resize((64, 64))).extrema[0]
            records.append(
                {
                    "image_id": str(row["image_id"]),
                    "class_name": str(row["class_name"]),
                    "split": str(row["audit_split"]),
                    "width": image.width,
                    "height": image.height,
                    "short_side": min(image.size),
                    "orientation": orientation,
                    "luma_range": int(luma_extrema[1] - luma_extrema[0]),
                    "perceptual_hash": file_difference_hash(image_path),
                }
            )
    images = pd.DataFrame(records)
    pairs: list[dict[str, object]] = []
    rows = images.to_dict("records")
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            distance = hamming_distance(
                str(first["perceptual_hash"]), str(second["perceptual_hash"])
            )
            if distance <= near_duplicate_distance:
                pairs.append(
                    {
                        "first_image_id": first["image_id"],
                        "first_class": first["class_name"],
                        "first_split": first["split"],
                        "second_image_id": second["image_id"],
                        "second_class": second["class_name"],
                        "second_split": second["split"],
                        "hamming_distance": distance,
                        "cross_split": first["split"] != second["split"],
                        "cross_class": first["class_name"] != second["class_name"],
                    }
                )
    pair_frame = pd.DataFrame(pairs)

    source_labels = (
        manifest.groupby(["class_name", "source_object_label"]).size().to_dict()
        if "source_object_label" in manifest.columns
        else {}
    )
    normal_rows = manifest[manifest["class_name"] == "normal"]
    normal_certified = (
        normal_rows.get("source_object_label", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.casefold()
        .isin({"normal", "undamaged", "no damage"})
    )
    dimensions = {
        class_name: {
            "count": int(len(group)),
            "short_side_min": int(group["short_side"].min()),
            "short_side_median": float(group["short_side"].median()),
            "short_side_max": int(group["short_side"].max()),
            "below_128_count": int((group["short_side"] < 128).sum()),
        }
        for class_name, group in images.groupby("class_name")
    }
    report: dict[str, object] = {
        "record_count": int(len(manifest)),
        "counts_by_class_and_split": {
            class_name: {split: int(value) for split, value in values.items()}
            for class_name, values in pd.crosstab(
                manifest["class_name"], manifest["audit_split"]
            ).to_dict(orient="index").items()
        },
        "source_label_counts": {
            f"{class_name}:{label}": int(count)
            for (class_name, label), count in source_labels.items()
        },
        "normal_rows_with_explicit_no_damage_label": int(normal_certified.sum()),
        "normal_rows_without_explicit_no_damage_label": int(
            len(normal_rows) - normal_certified.sum()
        ),
        "image_dimensions": dimensions,
        "non_default_exif_orientation_count": int((images["orientation"] != 1).sum()),
        "near_duplicate_threshold": near_duplicate_distance,
        "near_duplicate_pair_count": int(len(pair_frame)),
        "cross_split_near_duplicate_pair_count": int(
            pair_frame["cross_split"].sum() if not pair_frame.empty else 0
        ),
        "cross_class_near_duplicate_pair_count": int(
            pair_frame["cross_class"].sum() if not pair_frame.empty else 0
        ),
        "interpretation": [
            "A perceptual-hash match is a review candidate, not automatic proof of duplication.",
            "A body-part annotation does not certify that a panel is undamaged.",
            "Class-dependent crop size can become a shortcut unrelated to defect appearance.",
        ],
    }
    return report, pair_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-directory", type=Path, default=Path("data/manifests")
    )
    parser.add_argument("--images-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/audit/dataset_audit.json")
    )
    parser.add_argument(
        "--near-duplicate-output",
        type=Path,
        default=Path("artifacts/audit/near_duplicate_candidates.csv"),
    )
    parser.add_argument("--near-duplicate-distance", type=int, default=6)
    args = parser.parse_args()
    report, pairs = audit_dataset(
        args.manifest_directory, args.images_root, args.near_duplicate_distance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.near_duplicate_output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.near_duplicate_output, index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
