"""Exercise the live frontend, health, prediction, feedback, and OpenAPI routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/dataset_v2/processed/train/normal"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v2/metrics/live_api_verification.json"),
    )
    parser.add_argument("--skip-feedback", action="store_true")
    args = parser.parse_args()
    image_path = args.image
    if image_path.is_dir():
        image_path = next(
            path
            for path in sorted(image_path.iterdir())
            if path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        )
    base = args.api_url.rstrip("/")
    with httpx.Client(timeout=60.0) as client:
        root = client.get(f"{base}/")
        root.raise_for_status()
        health = client.get(f"{base}/health")
        health.raise_for_status()
        openapi = client.get(f"{base}/openapi.json")
        openapi.raise_for_status()
        media_type = "image/png" if image_path.suffix.casefold() == ".png" else "image/jpeg"
        with image_path.open("rb") as stream:
            prediction_response = client.post(
                f"{base}/api/v1/predict",
                files={"file": (image_path.name, stream, media_type)},
            )
        prediction_response.raise_for_status()
        prediction = prediction_response.json()
        predicted_class = str(prediction["predicted_class"])
        feedback_response = None
        if not args.skip_feedback:
            feedback_response = client.post(
                f"{base}/api/v1/feedback",
                json={
                    "prediction_id": prediction["prediction_id"],
                    "image_filename": image_path.name,
                    "image_sha256": prediction["image_sha256"],
                    "predicted_class": predicted_class,
                    "confidence": prediction["confidence"],
                    "is_correct": True,
                    "comment": "Automated professor-demonstration endpoint check.",
                },
            )
            feedback_response.raise_for_status()
    result = {
        "api_url": base,
        "root_status": root.status_code,
        "health_status": health.status_code,
        "health": health.json(),
        "openapi_status": openapi.status_code,
        "prediction_status": prediction_response.status_code,
        "prediction": prediction,
        "feedback_status": (
            feedback_response.status_code if feedback_response is not None else None
        ),
        "feedback": feedback_response.json() if feedback_response is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
