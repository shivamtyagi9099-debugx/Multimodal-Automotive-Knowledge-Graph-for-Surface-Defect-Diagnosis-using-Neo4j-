"""Shared image-orientation and duplicate-integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def normalized_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation once and return an independent RGB image."""
    return ImageOps.exif_transpose(image).convert("RGB")


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a deterministic perceptual dHash as lowercase hexadecimal.

    The hash is intentionally simple and dependency-free. It is used as a
    leakage alarm, not as proof that two photographs have the same semantic
    content. Candidate matches still require review.
    """
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    grayscale = normalized_rgb(image).convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale, dtype=np.int16)
    comparisons = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for index, bit in enumerate(comparisons):
        if bool(bit):
            value |= 1 << index
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def file_difference_hash(path: Path, hash_size: int = 8) -> str:
    """Decode one image and return its perceptual dHash."""
    with Image.open(path) as image:
        return difference_hash(image, hash_size)


def hamming_distance(first_hash: str, second_hash: str) -> int:
    """Return the bit distance between equal-width hexadecimal hashes."""
    first = first_hash.strip().casefold()
    second = second_hash.strip().casefold()
    if not first or len(first) != len(second):
        raise ValueError("Perceptual hashes must be non-empty and equal width")
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("Perceptual hashes must be hexadecimal") from exc


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
