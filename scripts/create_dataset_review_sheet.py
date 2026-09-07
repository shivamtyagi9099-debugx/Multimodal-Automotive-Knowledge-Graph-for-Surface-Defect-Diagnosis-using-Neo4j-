"""Create a deterministic labelled contact sheet for dataset quality review."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def create_review_sheet(
    manifest_path: Path,
    source_directory: Path,
    output_path: Path,
    samples_per_class: int = 6,
    random_seed: int = 42,
) -> None:
    """Render an equal, reproducible sample from every manifest class."""
    manifest = pd.read_csv(manifest_path)
    required = {"file_path", "class_name", "image_id"}
    missing = required - set(manifest.columns)
    if missing or manifest.empty:
        raise ValueError(f"Manifest is empty or missing columns: {sorted(missing)}")
    class_names = sorted(manifest["class_name"].astype(str).unique())
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    generator = random.Random(random_seed)
    tile_size = 224
    label_height = 34
    canvas = Image.new(
        "RGB",
        (samples_per_class * tile_size, len(class_names) * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for row_index, class_name in enumerate(class_names):
        records = manifest[manifest["class_name"] == class_name].to_dict("records")
        if len(records) < samples_per_class:
            raise ValueError(f"Not enough {class_name} records for the review sheet")
        selected = generator.sample(records, samples_per_class)
        for column_index, record in enumerate(selected):
            image_path = source_directory / str(record["file_path"])
            with Image.open(image_path) as source_image:
                tile = ImageOps.fit(
                    source_image.convert("RGB"),
                    (tile_size, tile_size),
                    method=Image.Resampling.LANCZOS,
                )
            x = column_index * tile_size
            y = row_index * (tile_size + label_height)
            canvas.paste(tile, (x, y))
            label = f"{class_name} | {record['image_id']}"
            draw.rectangle((x, y + tile_size, x + tile_size, y + tile_size + label_height), fill="white")
            draw.text((x + 5, y + tile_size + 7), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> None:
    """Parse command-line options and create the review artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/dataset_manifest.csv"),
    )
    parser.add_argument("--source-directory", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/figures/dataset_review_sheet.png"),
    )
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    create_review_sheet(
        args.manifest,
        args.source_directory,
        args.output,
        args.samples_per_class,
        args.random_seed,
    )
    print(f"Dataset review sheet written to {args.output}")


if __name__ == "__main__":
    main()
