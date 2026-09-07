"""Acquire a versioned, pre-inference candidate pool from licensed sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.data.image_integrity import normalized_rgb  # noqa: E402
from ml.src.data.import_huggingface_dataset import (  # noqa: E402
    DATASET_LICENSE,
    DATASET_NAME,
    DEFAULT_REPOSITORY,
    _raw_file_url,
    _request_bytes,
    _scaled_crop_box,
    collect_candidates,
)

PINNED_HITL_REVISION = "2dbf46f7e23eb0d6f551d38e4f77a11fe9d045b9"
COMMONS_CATEGORY = "Quality images of automobiles"
USER_AGENT = "PanelCheck-University-PoC/1.0"
CLASS_ORDER = ["normal", "scratch", "dent", "rust"]
RECORD_COLUMNS = [
    "file_path",
    "class_name",
    "original_label",
    "source_dataset",
    "source_url",
    "licence",
    "source_group",
    "review_status",
    "review_notes",
]


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value))).strip()


def _commons_candidates(
    limit: int,
    excluded_groups: set[str],
    category: str = COMMONS_CATEGORY,
) -> list[dict[str, str]]:
    base_parameters = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": str(min(limit + len(excluded_groups) + 50, 500)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "1280",
        "format": "json",
        "formatversion": "2",
    }
    records: list[dict[str, str]] = []
    seen_groups: set[str] = set()
    continuation: dict[str, str] = {}
    while len(records) < limit:
        parameters = {**base_parameters, **continuation}
        request = urllib.request.Request(
            "https://commons.wikimedia.org/w/api.php?"
            + urllib.parse.urlencode(parameters),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.load(response)
        for page in sorted(
            payload.get("query", {}).get("pages", []),
            key=lambda item: int(item["pageid"]),
        ):
            if not page.get("imageinfo"):
                continue
            group = f"commons-{page['pageid']}"
            if group in excluded_groups or group in seen_groups:
                continue
            information = page["imageinfo"][0]
            metadata = information.get("extmetadata", {})
            licence = str(metadata.get("LicenseShortName", {}).get("value", "")).strip()
            licence_url = str(metadata.get("LicenseUrl", {}).get("value", "")).strip()
            if not licence or not licence_url:
                continue
            records.append(
                {
                    "group": group,
                    "title": str(page["title"]),
                    "download_url": str(
                        information.get("thumburl") or information["url"]
                    ),
                    "source_url": str(information["descriptionurl"]),
                    "licence": f"{licence} ({licence_url})",
                    "artist": _plain_text(
                        str(
                            metadata.get("Artist", {}).get(
                                "value", "Wikimedia Commons contributor"
                            )
                        )
                    ),
                }
            )
            seen_groups.add(group)
            if len(records) == limit:
                break
        continuation = payload.get("continue", {})
        if not continuation:
            break
    if len(records) < limit:
        raise RuntimeError(f"Only {len(records)} Commons candidates were available")
    return records


def _save_commons_candidate(
    record: dict[str, str],
    raw_directory: Path,
    class_name: str = "normal",
    category: str = COMMONS_CATEGORY,
) -> dict[str, str] | None:
    try:
        request = urllib.request.Request(
            record["download_url"], headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            image = normalized_rgb(source)
        if min(image.size) < 256:
            return None
        relative_path = Path(class_name) / f"{record['group']}.jpg"
        output_path = raw_directory / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95, optimize=True)
        return {
            "file_path": relative_path.as_posix(),
            "class_name": class_name,
            "original_label": (
                "quality automobile photograph; requires no-damage review"
                if class_name == "normal"
                else f"{category}; requires body-panel corrosion review"
            ),
            "source_dataset": f"Wikimedia Commons / {category}",
            "source_url": record["source_url"],
            "licence": record["licence"],
            "source_group": record["group"],
            "review_status": "pending",
            "review_notes": f"Creator: {record['artist']}",
        }
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _save_hitl_candidate(candidate: object, raw_directory: Path) -> dict[str, str] | None:
    image_url = _raw_file_url(
        DEFAULT_REPOSITORY, PINNED_HITL_REVISION, candidate.image_path
    )
    try:
        data = _request_bytes(image_url, retries=2, timeout=60)
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            image = normalized_rgb(source)
        crop_box = _scaled_crop_box(candidate, image.width, image.height)
        crop = image.crop(crop_box)
        digest = hashlib.sha256(
            f"{PINNED_HITL_REVISION}:{candidate.source_group}".encode("utf-8")
        ).hexdigest()[:16]
        relative_path = Path(candidate.class_name) / f"hitl-{digest}.jpg"
        output_path = raw_directory / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="JPEG", quality=95, optimize=True)
        return {
            "file_path": relative_path.as_posix(),
            "class_name": candidate.class_name,
            "original_label": candidate.source_label,
            "source_dataset": DATASET_NAME,
            "source_url": image_url,
            "licence": DATASET_LICENSE,
            "source_group": candidate.source_group,
            "review_status": "pending",
            "review_notes": (
                f"Pinned mirror revision {PINNED_HITL_REVISION}; "
                f"annotation {candidate.annotation_path}; crop {candidate.crop_box}"
            ),
        }
    except (OSError, ValueError, RuntimeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory", type=Path, default=Path("data/dataset_v2/raw")
    )
    parser.add_argument(
        "--records", type=Path, default=Path("data/dataset_v2/source_records.csv")
    )
    parser.add_argument("--normal-candidates", type=int, default=130)
    parser.add_argument("--scratch-candidates", type=int, default=120)
    parser.add_argument("--dent-candidates", type=int, default=120)
    parser.add_argument("--rust-candidates", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--annotation-cache",
        type=Path,
        default=Path("data/cache/dataset_annotations"),
    )
    args = parser.parse_args()
    if args.records.is_file() and args.records.stat().st_size > len(
        ",".join(RECORD_COLUMNS)
    ) + 2:
        raise SystemExit(
            f"Candidate records already exist: {args.records}. Refusing to overwrite."
        )
    if any(
        path.is_file() and path.name != ".gitkeep"
        for path in args.raw_directory.rglob("*")
    ):
        raise SystemExit(
            f"Candidate images already exist under {args.raw_directory}. Refusing to overwrite."
        )

    demo_manifest = pd.read_csv(PROJECT_ROOT / "demo_images" / "demo_manifest.csv")
    excluded_groups = set(demo_manifest["source_group"].astype(str))
    baseline_test = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "test_manifest.csv")
    excluded_groups.update(baseline_test["source_group"].astype(str))

    commons = _commons_candidates(args.normal_candidates, excluded_groups)
    with ThreadPoolExecutor(max_workers=8) as executor:
        normal_rows = [
            row
            for row in executor.map(
                lambda record: _save_commons_candidate(record, args.raw_directory),
                commons,
            )
            if row is not None
        ]

    candidates = collect_candidates(
        DEFAULT_REPOSITORY,
        PINNED_HITL_REVISION,
        cache_directory=args.annotation_cache,
        request_workers=8,
    )
    generator = random.Random(args.random_seed)
    selected: list[object] = []
    requested = {
        "scratch": args.scratch_candidates,
        "dent": args.dent_candidates,
        "rust": args.rust_candidates,
    }
    for class_name, count in requested.items():
        available = sorted(
            [
                candidate
                for candidate in candidates
                if candidate.class_name == class_name
                and candidate.source_group not in excluded_groups
            ],
            key=lambda candidate: candidate.source_group,
        )
        generator.shuffle(available)
        # Prefer usable native crop resolution, with deterministic diversity
        # within each quality tier rather than choosing by model behavior.
        available.sort(
            key=lambda candidate: min(
                candidate.crop_box[2] - candidate.crop_box[0],
                candidate.crop_box[3] - candidate.crop_box[1],
            )
            < 128
        )
        if len(available) < count:
            raise RuntimeError(
                f"Requested {count} {class_name} candidates; only {len(available)} remain"
            )
        selected.extend(available[:count])
    with ThreadPoolExecutor(max_workers=8) as executor:
        defect_rows = [
            row
            for row in executor.map(
                lambda candidate: _save_hitl_candidate(candidate, args.raw_directory),
                selected,
            )
            if row is not None
        ]

    rows = normal_rows + defect_rows
    rows.sort(key=lambda row: (CLASS_ORDER.index(row["class_name"]), row["file_path"]))
    args.records.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    measured_counts = pd.DataFrame(rows)["class_name"].value_counts().to_dict()
    print(json.dumps({"candidate_counts": measured_counts}, indent=2))
    print("All rows remain review_status=pending; no model inference was run.")


if __name__ == "__main__":
    main()
