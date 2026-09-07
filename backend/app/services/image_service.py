"""Validation and safe local persistence for uploaded images."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

MEDIA_EXTENSIONS = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
}
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png"}


def calculate_image_sha256(file_bytes: bytes) -> str:
    """Return a stable fingerprint for matching an exact image to feedback."""
    if not file_bytes:
        raise ValueError("Uploaded image is empty")
    return hashlib.sha256(file_bytes).hexdigest()


def validate_uploaded_image(
    filename: str,
    content_type: str,
    size_bytes: int,
    allowed_image_types: list[str] | None = None,
    maximum_size_bytes: int | None = None,
) -> bool:
    """Validate upload metadata before decoding and storage."""
    if not filename or size_bytes <= 0:
        return False
    normalized_type = content_type.strip().lower()
    allowed = set(allowed_image_types or MEDIA_EXTENSIONS)
    if normalized_type not in allowed or normalized_type not in MEDIA_EXTENSIONS:
        return False
    if maximum_size_bytes is not None and size_bytes > maximum_size_bytes:
        return False
    return Path(filename).suffix.lower() in MEDIA_EXTENSIONS[normalized_type]


def save_uploaded_image(
    file_bytes: bytes,
    original_filename: str,
    upload_directory: Path,
) -> Path:
    """Verify image bytes and save them under a generated, safe filename."""
    if not file_bytes:
        raise ValueError("Uploaded image is empty")
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            image.verify()
            image_format = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded content is not a readable image") from exc
    if image_format not in FORMAT_EXTENSIONS:
        raise ValueError("Only JPEG and PNG image data is supported")

    # Decode a second time because ``verify`` deliberately invalidates the
    # first image object. This also ensures EXIF orientation is readable before
    # the original bytes are persisted and normalized during inference.
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            ImageOps.exif_transpose(image).load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded image cannot be fully decoded") from exc

    upload_directory.mkdir(parents=True, exist_ok=True)
    output_path = upload_directory / f"{uuid4().hex}{FORMAT_EXTENSIONS[image_format]}"
    output_path.write_bytes(file_bytes)
    return output_path
