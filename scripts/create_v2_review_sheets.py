"""Render every dataset-v2 candidate for pre-split human quality review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

CLASS_ORDER = ["normal", "scratch", "dent", "rust"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=Path("data/dataset_v2/source_records.csv")
    )
    parser.add_argument("--raw-directory", type=Path, default=Path("data/dataset_v2/raw"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/v2/review_sheets"),
    )
    parser.add_argument("--images-per-sheet", type=int, default=25)
    args = parser.parse_args()
    records = pd.read_csv(args.records, keep_default_na=False)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    columns = 5
    tile = 300
    label_height = 52
    rows_per_page = (args.images_per_sheet + columns - 1) // columns
    for class_name in CLASS_ORDER:
        class_records = records[records["class_name"] == class_name].to_dict("records")
        for page_index, start in enumerate(range(0, len(class_records), args.images_per_sheet), start=1):
            page_records = class_records[start : start + args.images_per_sheet]
            canvas = Image.new(
                "RGB", (columns * tile, rows_per_page * (tile + label_height)), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for local_index, record in enumerate(page_records):
                image_path = args.raw_directory / str(record["file_path"])
                with Image.open(image_path) as source:
                    source.load()
                    image = ImageOps.contain(
                        ImageOps.exif_transpose(source).convert("RGB"),
                        (tile, tile),
                        Image.Resampling.LANCZOS,
                    )
                x = local_index % columns * tile
                y = local_index // columns * (tile + label_height)
                canvas.paste(image, (x + (tile - image.width) // 2, y + (tile - image.height) // 2))
                record_number = start + local_index
                draw.text(
                    (x + 4, y + tile + 3),
                    f"{record_number:03d} | {Path(str(record['file_path'])).name[:34]}\n"
                    f"{record['original_label']}",
                    fill="black",
                )
            output = args.output_directory / f"{class_name}-{page_index:02d}.jpg"
            canvas.save(output, format="JPEG", quality=92)
            print(output)


if __name__ == "__main__":
    main()
