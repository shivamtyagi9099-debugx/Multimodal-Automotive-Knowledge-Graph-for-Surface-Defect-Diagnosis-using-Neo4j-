"""Tests for deterministic public-dataset label and crop admission rules."""

from ml.src.data.import_huggingface_dataset import (
    CropCandidate,
    candidate_from_annotation,
    select_balanced_candidates,
)


def _polygon(label: str, left: int, top: int, right: int, bottom: int) -> dict:
    """Create one rectangular Supervisely polygon fixture."""
    return {
        "classTitle": label,
        "geometryType": "polygon",
        "points": {
            "exterior": [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ]
        },
    }


def _annotation(*objects: dict) -> dict:
    """Create a valid image annotation fixture."""
    return {"size": {"width": 800, "height": 600}, "objects": list(objects)}


def test_damage_labels_map_to_scoped_classes() -> None:
    """Source labels should map only to scratch, dent, and rust."""
    cases = (("Scratch", "scratch"), ("Dent", "dent"), ("Corrosion", "rust"))
    for source_label, expected_class in cases:
        candidate = candidate_from_annotation(
            _annotation(_polygon(source_label, 200, 150, 400, 350)),
            "Car parts dataset/File1/ann/example.jpg.json",
            "damage",
        )
        assert candidate is not None
        assert candidate.class_name == expected_class
        assert candidate.image_path.endswith("/img/example.jpg")


def test_ambiguous_overlapping_damage_is_excluded() -> None:
    """A target overlapped by a different defect should not receive one label."""
    candidate = candidate_from_annotation(
        _annotation(
            _polygon("Scratch", 100, 100, 400, 400),
            _polygon("Dent", 120, 120, 380, 380),
        ),
        "Car parts dataset/File1/ann/ambiguous.jpg.json",
        "damage",
    )
    assert candidate is None


def test_normal_crop_uses_largest_body_panel() -> None:
    """Normal examples should be derived from an explicitly annotated panel."""
    candidate = candidate_from_annotation(
        _annotation(
            _polygon("wheel", 10, 10, 150, 150),
            _polygon("front-door", 200, 100, 650, 500),
        ),
        "Car damages dataset/File1/ann/panel.jpg.json",
        "normal",
    )
    assert candidate is not None
    assert candidate.class_name == "normal"
    assert candidate.source_label == "front-door"


def test_balanced_selection_is_reproducible_and_exact() -> None:
    """The importer should select an equal multiple-of-ten count per class."""
    candidates = []
    for class_name in ("normal", "scratch", "dent", "rust"):
        for index in range(20):
            source_group = f"{class_name}-{index}.jpg"
            candidates.append(
                CropCandidate(
                    class_name=class_name,
                    source_label=class_name,
                    annotation_path=f"ann/{source_group}.json",
                    image_path=f"img/{source_group}",
                    source_group=source_group,
                    crop_box=(0, 0, 128, 128),
                    annotation_width=128,
                    annotation_height=128,
                )
            )

    first = select_balanced_candidates(candidates, 10, 100, 42)
    second = select_balanced_candidates(candidates, 10, 100, 42)
    assert first == second
    assert len(first) == 40
    assert {
        name: sum(candidate.class_name == name for candidate in first)
        for name in ("normal", "scratch", "dent", "rust")
    } == {"normal": 10, "scratch": 10, "dent": 10, "rust": 10}
    assert len({candidate.source_group for candidate in first}) == len(first)
