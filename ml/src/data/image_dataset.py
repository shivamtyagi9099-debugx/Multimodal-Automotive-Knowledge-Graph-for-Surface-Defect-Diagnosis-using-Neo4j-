"""Ordered ImageFolder dataset and shared PyTorch image transforms."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ml.src.data.image_integrity import normalized_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SquarePad:
    """Pad an image to a square without discarding or stretching evidence."""

    def __init__(self, fill: tuple[int, int, int] = (124, 116, 104)) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        horizontal = side - width
        vertical = side - height
        return ImageOps.expand(
            image,
            border=(
                horizontal // 2,
                vertical // 2,
                horizontal - horizontal // 2,
                vertical - vertical // 2,
            ),
            fill=self.fill,
        )


def load_oriented_rgb(path: str) -> Image.Image:
    """Load an image with EXIF orientation applied before RGB conversion."""
    with Image.open(path) as image:
        image.load()
        return normalized_rgb(image)


class OrderedImageFolder(ImageFolder):
    """ImageFolder that preserves the project's configured class order."""

    def __init__(
        self,
        root: str | Path,
        class_names: list[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.ordered_class_names = list(class_names)
        if len(self.ordered_class_names) < 2:
            raise ValueError("At least two ordered class names are required")
        if len(set(self.ordered_class_names)) != len(self.ordered_class_names):
            raise ValueError("Class names must be unique")
        super().__init__(
            root=str(root),
            transform=transform,
            loader=load_oriented_rgb,
        )

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """Return configured class names after validating directory coverage."""
        root = Path(directory)
        expected = set(self.ordered_class_names)
        observed = {path.name for path in root.iterdir() if path.is_dir()}
        missing = expected - observed
        unexpected = observed - expected
        if missing:
            raise FileNotFoundError(f"Missing class directories: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"Unexpected class directories: {sorted(unexpected)}")
        return self.ordered_class_names, {
            class_name: index
            for index, class_name in enumerate(self.ordered_class_names)
        }


def build_training_transform(
    image_size: int,
    geometry: str = "square_pad_then_resize",
) -> transforms.Compose:
    """Return evidence-preserving augmentation plus ImageNet normalization."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if geometry not in {"square_pad_then_resize", "resize_stretch"}:
        raise ValueError(f"Unsupported preprocessing geometry: {geometry}")
    geometry_transforms: list[object] = (
        [SquarePad(), transforms.Resize((image_size, image_size))]
        if geometry == "square_pad_then_resize"
        else [transforms.Resize((image_size, image_size))]
    )
    return transforms.Compose(
        [
            *geometry_transforms,
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=6,
                translate=(0.025, 0.025),
                scale=(0.95, 1.05),
                fill=(124, 116, 104),
            ),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_evaluation_transform(
    image_size: int,
    geometry: str = "square_pad_then_resize",
) -> transforms.Compose:
    """Return deterministic resize and ImageNet normalization."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if geometry not in {"square_pad_then_resize", "resize_stretch"}:
        raise ValueError(f"Unsupported preprocessing geometry: {geometry}")
    geometry_transforms: list[object] = (
        [SquarePad(), transforms.Resize((image_size, image_size))]
        if geometry == "square_pad_then_resize"
        else [transforms.Resize((image_size, image_size))]
    )
    return transforms.Compose(
        [
            *geometry_transforms,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_image_loader(
    directory: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    training: bool,
    num_workers: int = 0,
    shuffle: bool | None = None,
    preprocessing_geometry: str = "square_pad_then_resize",
) -> DataLoader:
    """Create an ordered PyTorch DataLoader for one dataset split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")
    transform = (
        build_training_transform(image_size, preprocessing_geometry)
        if training
        else build_evaluation_transform(image_size, preprocessing_geometry)
    )
    dataset = OrderedImageFolder(directory, class_names, transform=transform)
    should_shuffle = training if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
