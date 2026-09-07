"""Reproduce the pre-inference, licence-traceable presentation image set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.data.image_integrity import (  # noqa: E402
    difference_hash,
    file_difference_hash,
    file_sha256,
    hamming_distance,
    normalized_rgb,
)
from ml.src.data.import_huggingface_dataset import (  # noqa: E402
    DATASET_LICENSE,
    DATASET_NAME,
    DATASET_PAGE,
    DEFAULT_REPOSITORY,
    _raw_file_url,
    _request_bytes,
    _scaled_crop_box,
    collect_candidates,
)

CLASS_ORDER = ["normal", "scratch", "dent", "rust"]
PINNED_HITL_REVISION = "2dbf46f7e23eb0d6f551d38e4f77a11fe9d045b9"
SELECTION_PROTOCOL = "panelcheck-demo-visual-review-v1-2026-09-06"
COMMONS_PAGE_IDS = {
    "normal": [
        13249706,
        13250815,
        13266809,
        13285156,
        148482436,
        163247508,
        163259094,
        163277205,
        171254346,
        175112917,
    ],
    "rust": [
        6812018,
        18484033,
        19820877,
        38584707,
        38911657,
        39677503,
        61972729,
        88806007,
        129623127,
        174158159,
    ],
}
HITL_SOURCE_IMAGES = {
    "scratch": [
        "Car parts dataset/File1/img/Car damages 254.png",
        "Car parts dataset/File1/img/Car damages 299.png",
        "Car parts dataset/File1/img/Car damages 798.png",
        "Car parts dataset/File1/img/Car damages 1327.png",
        "Car parts dataset/File1/img/Car damages 395.png",
        "Car parts dataset/File1/img/Car damages 222.png",
        "Car parts dataset/File1/img/Car damages 422.png",
        "Car parts dataset/File1/img/Car damages 418.png",
        "Car parts dataset/File1/img/Car damages 1346.png",
        "Car parts dataset/File1/img/Car damages 637.png",
    ],
    "dent": [
        "Car parts dataset/File1/img/Car damages 688.png",
        "Car parts dataset/File1/img/Car damages 702.png",
        "Car parts dataset/File1/img/Car damages 397.png",
        "Car parts dataset/File1/img/Car damages 1329.png",
        "Car parts dataset/File1/img/Car damages 983.png",
        "Car parts dataset/File1/img/Car damages 1230.png",
        "Car parts dataset/File1/img/Car damages 804.png",
        "Car parts dataset/File1/img/Car damages 196.png",
        "Car parts dataset/File1/img/Car damages 257.png",
        "Car parts dataset/File1/img/Car damages 786.png",
    ],
}
USER_AGENT = "PanelCheck-University-PoC/1.0"


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value))).strip()


def _commons_metadata(page_ids: list[int]) -> list[dict[str, str]]:
    parameters = {
        "action": "query",
        "pageids": "|".join(str(value) for value in page_ids),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "1280",
        "format": "json",
        "formatversion": "2",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
        parameters
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    pages = {int(page["pageid"]): page for page in payload["query"]["pages"]}
    records: list[dict[str, str]] = []
    for page_id in page_ids:
        page = pages[page_id]
        information = page["imageinfo"][0]
        metadata = information.get("extmetadata", {})
        licence = str(metadata.get("LicenseShortName", {}).get("value", "")).strip()
        licence_url = str(metadata.get("LicenseUrl", {}).get("value", "")).strip()
        if not licence or not licence_url:
            raise ValueError(f"Commons page {page_id} has incomplete licence metadata")
        records.append(
            {
                "page_id": str(page_id),
                "title": str(page["title"]),
                "download_url": str(information.get("thumburl") or information["url"]),
                "source_url": str(information["descriptionurl"]),
                "licence": licence,
                "licence_url": licence_url,
                "attribution": _plain_text(
                    str(metadata.get("Artist", {}).get("value", "Wikimedia Commons contributor"))
                ),
            }
        )
    return records


def _download_commons(
    class_name: str,
    record: dict[str, str],
    output_directory: Path,
) -> dict[str, str]:
    request = urllib.request.Request(
        record["download_url"], headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        image_bytes = response.read()
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        image = normalized_rgb(source)
    stable_id = f"commons-{record['page_id']}"
    output_path = output_directory / class_name / f"{stable_id}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=95, optimize=True)
    return {
        "filename": output_path.relative_to(output_directory).as_posix(),
        "expected_class": class_name,
        "original_label": "visually reviewed undamaged car" if class_name == "normal" else "rusty automobile",
        "source_dataset": "Wikimedia Commons",
        "source_url": record["source_url"],
        "licence": record["licence"],
        "licence_url": record["licence_url"],
        "attribution": record["attribution"],
        "source_group": stable_id,
    }


def _download_hitl(
    candidate: object,
    output_directory: Path,
) -> dict[str, str]:
    image_url = _raw_file_url(
        DEFAULT_REPOSITORY, PINNED_HITL_REVISION, candidate.image_path
    )
    image_bytes = _request_bytes(image_url, retries=2, timeout=60)
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        image = normalized_rgb(source)
    crop_box = _scaled_crop_box(candidate, image.width, image.height)
    crop = image.crop(crop_box)
    stable_id = hashlib.sha256(
        f"{PINNED_HITL_REVISION}:{candidate.source_group}".encode("utf-8")
    ).hexdigest()[:16]
    output_path = output_directory / candidate.class_name / f"hitl-{stable_id}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="JPEG", quality=95, optimize=True)
    return {
        "filename": output_path.relative_to(output_directory).as_posix(),
        "expected_class": candidate.class_name,
        "original_label": candidate.source_label,
        "source_dataset": DATASET_NAME,
        "source_url": image_url,
        "licence": DATASET_LICENSE,
        "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": "Humans in the Loop / CC0 1.0",
        "source_group": candidate.source_group,
    }


def _verify_separation(rows: list[dict[str, str]], output_directory: Path) -> None:
    development = pd.concat(
        [
            pd.read_csv(PROJECT_ROOT / "data" / "manifests" / f"{split}_manifest.csv")
            for split in ("train", "validation", "test")
        ],
        ignore_index=True,
    )
    development_groups = set(development.get("source_group", pd.Series(dtype=str)).astype(str))
    development_checksums = set(development["checksum"].astype(str).str.casefold())
    development_hashes: list[tuple[str, str]] = []
    for _, record in development.iterrows():
        path = PROJECT_ROOT / "data" / "raw" / str(record["file_path"])
        development_hashes.append((str(record["image_id"]), file_difference_hash(path)))

    demo_hashes: list[tuple[str, str]] = []
    for row in rows:
        path = output_directory / row["filename"]
        row["checksum"] = file_sha256(path)
        row["perceptual_hash"] = file_difference_hash(path)
        row["excluded_from_training_validation"] = "yes"
        row["selected_before_inference"] = "yes"
        row["selection_protocol_version"] = SELECTION_PROTOCOL
        if row["source_group"] in development_groups:
            raise ValueError(f"Demo source group leaks from development: {row['source_group']}")
        if row["checksum"] in development_checksums:
            raise ValueError(f"Demo checksum leaks from development: {row['filename']}")
        for development_id, development_hash in development_hashes:
            distance = hamming_distance(row["perceptual_hash"], development_hash)
            if distance <= 6:
                raise ValueError(
                    f"Demo image {row['filename']} is perceptually close to "
                    f"development image {development_id} (distance {distance})"
                )
        demo_hashes.append((row["filename"], row["perceptual_hash"]))
    for index, (first_name, first_hash) in enumerate(demo_hashes):
        for second_name, second_hash in demo_hashes[index + 1 :]:
            distance = hamming_distance(first_hash, second_hash)
            if distance <= 6:
                raise ValueError(
                    f"Demo near-duplicate pair: {first_name}, {second_name} "
                    f"(distance {distance})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("demo_images"))
    parser.add_argument(
        "--annotation-cache",
        type=Path,
        default=Path("data/cache/dataset_annotations"),
    )
    args = parser.parse_args()
    output_directory = args.output_directory
    existing_images = list(output_directory.glob("*/*")) if output_directory.exists() else []
    if existing_images:
        raise SystemExit(
            f"Demo output is not empty: {output_directory}. Use a new directory "
            "to preserve the frozen selection."
        )

    rows: list[dict[str, str]] = []
    for class_name, page_ids in COMMONS_PAGE_IDS.items():
        records = _commons_metadata(page_ids)
        with ThreadPoolExecutor(max_workers=6) as executor:
            rows.extend(
                executor.map(
                    lambda record: _download_commons(
                        class_name, record, output_directory
                    ),
                    records,
                )
            )

    candidates = collect_candidates(
        DEFAULT_REPOSITORY,
        PINNED_HITL_REVISION,
        cache_directory=args.annotation_cache,
        request_workers=8,
    )
    candidate_lookup = {candidate.image_path: candidate for candidate in candidates}
    selected_candidates = [
        candidate_lookup[path]
        for class_name in ("scratch", "dent")
        for path in HITL_SOURCE_IMAGES[class_name]
    ]
    with ThreadPoolExecutor(max_workers=6) as executor:
        rows.extend(
            executor.map(
                lambda candidate: _download_hitl(candidate, output_directory),
                selected_candidates,
            )
        )

    _verify_separation(rows, output_directory)
    rows.sort(key=lambda row: (CLASS_ORDER.index(row["expected_class"]), row["filename"]))
    manifest_path = output_directory / "demo_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Frozen {len(rows)} pre-inference demo images in {output_directory}")


if __name__ == "__main__":
    main()
