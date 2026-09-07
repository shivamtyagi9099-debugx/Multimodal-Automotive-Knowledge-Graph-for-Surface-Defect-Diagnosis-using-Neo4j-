"""Build a traceable four-class crop dataset from a pinned public mirror.

The underlying Car Parts and Car Damages Dataset is published by Humans in the
Loop under CC0 1.0. ``DrBimmer/car-parts-and-damage-dataset`` is a convenient
Hugging Face mirror pinned by commit SHA. The mirror currently advertises MIT,
but the canonical publisher's CC0 statement is the provenance authority used
by new manifests. This importer uses Supervisely polygons only to derive one
classification crop per source image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from ml.src.data.image_integrity import difference_hash, normalized_rgb

DEFAULT_REPOSITORY = "DrBimmer/car-parts-and-damage-dataset"
DEFAULT_REVISION = "main"
DATASET_LICENSE = "CC0-1.0"
DATASET_PAGE = (
    "https://humansintheloop.org/resources/datasets/"
    "car-parts-and-car-damages-dataset/"
)
DATASET_NAME = "Humans in the Loop Car Parts and Car Damages Dataset"

# The upstream repository's directory names are inverted: the directory named
# "Car parts dataset" contains damage annotations and vice versa.
DAMAGE_ANNOTATION_DIRECTORY = "Car parts dataset/File1/ann"
NORMAL_ANNOTATION_DIRECTORY = "Car damages dataset/File1/ann"

DAMAGE_LABEL_MAP = {
    "dent": "dent",
    "scratch": "scratch",
    "corrosion": "rust",
}
PANEL_LABELS = {
    "back-bumper",
    "back-door",
    "fender",
    "front-bumper",
    "front-door",
    "hood",
    "quarter-panel",
    "rocker-panel",
    "roof",
    "trunk",
}
MINIMUM_CROP_DIMENSION = 96
MANIFEST_COLUMNS = [
    "image_id",
    "file_path",
    "class_name",
    "source",
    "dataset_split",
    "license",
    "checksum",
    "original_label",
    "source_dataset",
    "source_url",
    "licence",
    "perceptual_hash",
    "source_repository",
    "source_revision",
    "source_image_path",
    "source_annotation_path",
    "source_object_label",
    "source_group",
    "crop_box",
]


@dataclass(frozen=True)
class CropCandidate:
    """Describe one deterministic classification crop before image download."""

    class_name: str
    source_label: str
    annotation_path: str
    image_path: str
    source_group: str
    crop_box: tuple[int, int, int, int]
    annotation_width: int
    annotation_height: int


def _request_bytes(url: str, retries: int = 3, timeout: int = 45) -> bytes:
    """Download public bytes with bounded retries and a clear user agent."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError("Dataset downloads are restricted to huggingface.co HTTPS URLs")
    request = Request(url, headers={"User-Agent": "PanelCheck-University-PoC/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Dataset request failed after {retries} attempts: {url}") from last_error


def resolve_repository_revision(repository: str, revision: str) -> str:
    """Resolve a moving branch name to an immutable repository commit SHA."""
    repository_path = quote(repository, safe="/")
    metadata_url = f"https://huggingface.co/api/datasets/{repository_path}/revision/{quote(revision, safe='')}"
    metadata = json.loads(_request_bytes(metadata_url))
    resolved = str(metadata.get("sha", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Hugging Face did not return a valid dataset revision SHA")
    return resolved


def list_annotation_paths(
    repository: str,
    revision: str,
    directory: str,
) -> list[str]:
    """List every JSON annotation in one repository directory."""
    repository_path = quote(repository, safe="/")
    revision_path = quote(revision, safe="")
    directory_path = quote(directory, safe="/")
    api_url = (
        f"https://huggingface.co/api/datasets/{repository_path}/tree/"
        f"{revision_path}/{directory_path}?limit=1000&expand=false"
    )
    entries = json.loads(_request_bytes(api_url))
    if not isinstance(entries, list):
        raise ValueError("Hugging Face directory listing was not a JSON list")
    paths = sorted(
        str(entry["path"])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("type") == "file"
        and str(entry.get("path", "")).lower().endswith(".json")
    )
    if not paths:
        raise ValueError(f"No annotation JSON files found in {directory}")
    if len(paths) >= 1000:
        raise ValueError(
            f"Annotation listing for {directory} reached the API page limit; "
            "pagination support is required before using this revision"
        )
    return paths


def _raw_file_url(repository: str, revision: str, file_path: str) -> str:
    """Return the immutable raw-file URL for one repository path."""
    return (
        "https://huggingface.co/datasets/"
        f"{quote(repository, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(file_path, safe='/')}"
    )


def _image_path_from_annotation(annotation_path: str) -> str:
    """Convert the source's ``ann/image.ext.json`` path to ``img/image.ext``."""
    marker = "/ann/"
    if marker not in annotation_path or not annotation_path.endswith(".json"):
        raise ValueError(f"Unexpected annotation path: {annotation_path}")
    return annotation_path.replace(marker, "/img/", 1)[:-5]


def _normalized_objects(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return polygon objects with usable labels and point coordinates."""
    objects = annotation.get("objects")
    if not isinstance(objects, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict) or item.get("geometryType") != "polygon":
            continue
        title = str(item.get("classTitle", "")).strip()
        exterior = item.get("points", {}).get("exterior")
        if not title or not isinstance(exterior, list) or len(exterior) < 3:
            continue
        points: list[tuple[float, float]] = []
        for point in exterior:
            if (
                isinstance(point, list)
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                points.append((float(point[0]), float(point[1])))
        if len(points) >= 3:
            normalized.append({"title": title, "points": points})
    return normalized


def _bounding_box(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Return the axis-aligned box containing one polygon."""
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _box_area(box: tuple[float, float, float, float]) -> float:
    """Return the non-negative area of an axis-aligned box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_fraction(
    target: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
) -> float:
    """Return intersection area as a fraction of the target box."""
    target_area = _box_area(target)
    if target_area <= 0:
        return 1.0
    intersection = (
        max(0.0, min(target[2], other[2]) - max(target[0], other[0]))
        * max(0.0, min(target[3], other[3]) - max(target[1], other[1]))
    )
    return intersection / target_area


def _square_context_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin_ratio: float = 0.20,
) -> tuple[int, int, int, int] | None:
    """Expand a polygon box into a clamped square crop with visual context."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    side = max(width, height) * (1.0 + 2.0 * margin_ratio)
    if side < MINIMUM_CROP_DIMENSION:
        side = float(MINIMUM_CROP_DIMENSION)
    side = min(side, float(image_width), float(image_height))
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    left = max(0.0, min(center_x - side / 2.0, image_width - side))
    top = max(0.0, min(center_y - side / 2.0, image_height - side))
    result = (
        int(round(left)),
        int(round(top)),
        int(round(left + side)),
        int(round(top + side)),
    )
    if result[2] - result[0] < MINIMUM_CROP_DIMENSION:
        return None
    if result[3] - result[1] < MINIMUM_CROP_DIMENSION:
        return None
    return result


def candidate_from_annotation(
    annotation: dict[str, Any],
    annotation_path: str,
    source_kind: str,
) -> CropCandidate | None:
    """Select one unambiguous target crop from a source annotation."""
    size = annotation.get("size")
    if not isinstance(size, dict):
        return None
    try:
        image_width = int(size["width"])
        image_height = int(size["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if image_width < MINIMUM_CROP_DIMENSION or image_height < MINIMUM_CROP_DIMENSION:
        return None

    objects = _normalized_objects(annotation)
    if not objects:
        return None
    selected: dict[str, Any] | None = None
    selected_class: str | None = None

    if source_kind == "damage":
        damage_objects = [
            {
                **item,
                "normalized_title": item["title"].strip().casefold(),
                "box": _bounding_box(item["points"]),
            }
            for item in objects
        ]
        eligible: list[tuple[float, dict[str, Any], str]] = []
        for item in damage_objects:
            target_class = DAMAGE_LABEL_MAP.get(item["normalized_title"])
            if target_class is None:
                continue
            overlap = max(
                (
                    _intersection_fraction(item["box"], other["box"])
                    for other in damage_objects
                    if other is not item
                    and other["normalized_title"] != item["normalized_title"]
                ),
                default=0.0,
            )
            if overlap <= 0.10:
                eligible.append((_box_area(item["box"]), item, target_class))
        if eligible:
            _, selected, selected_class = max(eligible, key=lambda value: value[0])
    elif source_kind == "normal":
        eligible_parts = [
            item
            for item in objects
            if item["title"].strip().casefold() in PANEL_LABELS
        ]
        if eligible_parts:
            selected = max(
                eligible_parts,
                key=lambda item: _box_area(_bounding_box(item["points"])),
            )
            selected_class = "normal"
            selected = {**selected, "box": _bounding_box(selected["points"])}
    else:
        raise ValueError("source_kind must be 'damage' or 'normal'")

    if selected is None or selected_class is None:
        return None
    crop_box = _square_context_box(
        selected["box"],
        image_width,
        image_height,
    )
    if crop_box is None:
        return None
    image_path = _image_path_from_annotation(annotation_path)
    return CropCandidate(
        class_name=selected_class,
        source_label=str(selected["title"]),
        annotation_path=annotation_path,
        image_path=image_path,
        source_group=image_path,
        crop_box=crop_box,
        annotation_width=image_width,
        annotation_height=image_height,
    )


def collect_candidates(
    repository: str,
    revision: str,
    cache_directory: Path | None = None,
    request_workers: int = 8,
) -> list[CropCandidate]:
    """Download/cache annotations and build all eligible crop candidates."""
    if request_workers <= 0:
        raise ValueError("request_workers must be positive")
    candidates: list[CropCandidate] = []
    directories = (
        (DAMAGE_ANNOTATION_DIRECTORY, "damage"),
        (NORMAL_ANNOTATION_DIRECTORY, "normal"),
    )
    for directory, source_kind in directories:
        paths = list_annotation_paths(repository, revision, directory)
        print(f"Reviewing {len(paths)} {source_kind} annotation files...")

        def load_candidate(annotation_path: str) -> CropCandidate | None:
            """Load one cached or remote annotation and apply admission rules."""
            cache_path: Path | None = None
            annotation: Any = None
            if cache_directory is not None:
                cache_key = hashlib.sha256(
                    annotation_path.encode("utf-8")
                ).hexdigest()
                cache_path = cache_directory / revision / f"{cache_key}.json"
                if cache_path.is_file():
                    try:
                        annotation = json.loads(cache_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        annotation = None
            if annotation is None:
                annotation_bytes = _request_bytes(
                    _raw_file_url(repository, revision, annotation_path)
                )
                annotation = json.loads(annotation_bytes)
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = cache_path.with_suffix(".tmp")
                    temporary_path.write_bytes(annotation_bytes)
                    temporary_path.replace(cache_path)
            if isinstance(annotation, dict):
                return candidate_from_annotation(
                    annotation,
                    annotation_path,
                    source_kind,
                )
            return None

        with ThreadPoolExecutor(max_workers=request_workers) as executor:
            for index, candidate in enumerate(
                executor.map(load_candidate, paths), start=1
            ):
                if candidate is not None:
                    candidates.append(candidate)
                if index % 100 == 0:
                    print(f"  reviewed {index}/{len(paths)} annotations")
    return candidates


def select_balanced_candidates(
    candidates: list[CropCandidate],
    samples_per_class: int,
    maximum_samples_per_class: int,
    random_seed: int,
) -> list[CropCandidate]:
    """Select a deterministic balanced count compatible with 80/10/10."""
    by_class: dict[str, list[CropCandidate]] = {
        class_name: [] for class_name in ("normal", "scratch", "dent", "rust")
    }
    for candidate in candidates:
        if candidate.class_name in by_class:
            by_class[candidate.class_name].append(candidate)
    counts = {name: len(values) for name, values in by_class.items()}
    if min(counts.values(), default=0) < 10:
        raise ValueError(f"Insufficient eligible candidates for strict splitting: {counts}")

    if samples_per_class > 0:
        target_count = samples_per_class
    else:
        target_count = min(maximum_samples_per_class, min(counts.values()))
        target_count -= target_count % 10
    if target_count < 10 or target_count % 10 != 0:
        raise ValueError("samples_per_class must be a positive multiple of 10")
    insufficient = {
        name: count for name, count in counts.items() if count < target_count
    }
    if insufficient:
        raise ValueError(
            f"Requested {target_count} samples per class but candidates are {counts}"
        )

    selected: list[CropCandidate] = []
    generator = random.Random(random_seed)
    for class_name, values in by_class.items():
        ordered = sorted(values, key=lambda item: item.source_group)
        generator.shuffle(ordered)
        selected.extend(ordered[:target_count])
    return sorted(selected, key=lambda item: (item.class_name, item.source_group))


def _scaled_crop_box(
    candidate: CropCandidate,
    actual_width: int,
    actual_height: int,
) -> tuple[int, int, int, int]:
    """Scale annotation coordinates when decoded image dimensions differ."""
    scale_x = actual_width / candidate.annotation_width
    scale_y = actual_height / candidate.annotation_height
    left, top, right, bottom = candidate.crop_box
    return (
        max(0, int(round(left * scale_x))),
        max(0, int(round(top * scale_y))),
        min(actual_width, int(round(right * scale_x))),
        min(actual_height, int(round(bottom * scale_y))),
    )


def materialize_candidates(
    candidates: list[CropCandidate],
    repository: str,
    revision: str,
    output_directory: Path,
    manifest_path: Path,
) -> None:
    """Download selected images, save classification crops, and write manifest."""
    rows: list[dict[str, str]] = []
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates, start=1):
        image_url = _raw_file_url(repository, revision, candidate.image_path)
        image_bytes = _request_bytes(image_url)
        try:
            with Image.open(io.BytesIO(image_bytes)) as source_image:
                source_image.load()
                rgb_image = normalized_rgb(source_image)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(
                f"Source image could not be decoded: {candidate.image_path}"
            ) from exc

        crop_box = _scaled_crop_box(candidate, rgb_image.width, rgb_image.height)
        crop = rgb_image.crop(crop_box)
        stable_digest = hashlib.sha256(
            f"{revision}:{candidate.source_group}".encode("utf-8")
        ).hexdigest()[:16]
        image_id = f"{candidate.class_name}-{stable_digest}"
        relative_path = Path("classification") / candidate.class_name / f"{image_id}.jpg"
        output_path = output_directory / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="JPEG", quality=95, optimize=True)
        checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
        rows.append(
            {
                "image_id": image_id,
                "file_path": relative_path.as_posix(),
                "class_name": candidate.class_name,
                "source": DATASET_PAGE,
                "dataset_split": "",
                "license": DATASET_LICENSE,
                "checksum": checksum,
                "original_label": candidate.source_label,
                "source_dataset": DATASET_NAME,
                "source_url": image_url,
                "licence": DATASET_LICENSE,
                "perceptual_hash": difference_hash(crop),
                "source_repository": repository,
                "source_revision": revision,
                "source_image_path": candidate.image_path,
                "source_annotation_path": candidate.annotation_path,
                "source_object_label": candidate.source_label,
                "source_group": candidate.source_group,
                "crop_box": json.dumps(crop_box, separators=(",", ":")),
            }
        )
        if index % 25 == 0 or index == len(candidates):
            print(f"Downloaded and cropped {index}/{len(candidates)} images")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line options for the public dataset importer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-directory", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Balanced count (multiple of 10); 0 chooses automatically.",
    )
    parser.add_argument("--maximum-samples-per-class", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--annotation-cache",
        type=Path,
        default=Path("data/cache/dataset_annotations"),
    )
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Report eligible annotation counts without downloading images.",
    )
    return parser


def main() -> None:
    """Resolve the source version and create a balanced classification dataset."""
    args = build_parser().parse_args()
    if args.maximum_samples_per_class < 10:
        raise SystemExit("--maximum-samples-per-class must be at least 10")
    resolved_revision = resolve_repository_revision(args.repository, args.revision)
    print(f"Pinned dataset revision: {resolved_revision}")
    candidates = collect_candidates(
        args.repository,
        resolved_revision,
        cache_directory=args.annotation_cache,
        request_workers=args.request_workers,
    )
    counts = {
        class_name: sum(item.class_name == class_name for item in candidates)
        for class_name in ("normal", "scratch", "dent", "rust")
    }
    print(f"Eligible candidates: {json.dumps(counts, sort_keys=True)}")
    if args.inventory_only:
        return
    selected = select_balanced_candidates(
        candidates,
        samples_per_class=args.samples_per_class,
        maximum_samples_per_class=args.maximum_samples_per_class,
        random_seed=args.random_seed,
    )
    materialize_candidates(
        selected,
        args.repository,
        resolved_revision,
        args.output_directory,
        args.manifest,
    )
    print(f"Dataset manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
