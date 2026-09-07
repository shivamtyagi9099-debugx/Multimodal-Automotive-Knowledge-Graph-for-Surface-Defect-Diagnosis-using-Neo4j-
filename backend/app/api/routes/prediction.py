"""Image upload and classification endpoint."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.inventory import DefectCatalogEntry
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.catalog_service import (
    get_defect_details,
    get_inventory_last_updated,
)
from backend.app.services.image_service import (
    calculate_image_sha256,
    save_uploaded_image,
    validate_uploaded_image,
)
from backend.app.services.feedback_service import find_feedback_class
from backend.app.services.inference_service import preprocess_image, predict_class
from backend.app.services.neo4j_service import Neo4jServiceError

router = APIRouter(prefix="/api/v1", tags=["prediction"])

POC_DISCLAIMER = (
    "Academic proof-of-concept only. This is not a universal vehicle "
    "diagnostic tool."
)


async def _load_defect_knowledge(
    request: Request,
    class_name: str,
) -> tuple[DefectCatalogEntry | None, str, str]:
    """Prefer Neo4j data and transparently fall back to the static catalog."""
    catalog = request.app.state.catalog
    neo4j_service = request.app.state.neo4j_service
    if neo4j_service is not None:
        try:
            defect = await run_in_threadpool(
                neo4j_service.get_defect_details,
                class_name,
            )
            inventory_date = await run_in_threadpool(
                neo4j_service.get_inventory_last_updated
            )
        except Neo4jServiceError as exc:
            request.app.state.neo4j_connected = False
            request.app.state.neo4j_message = str(exc)
        else:
            if defect is not None:
                request.app.state.neo4j_connected = True
                request.app.state.neo4j_message = (
                    "Neo4j supplied repair and simulated inventory data."
                )
                return defect, inventory_date, "neo4j"
            request.app.state.neo4j_connected = False
            request.app.state.neo4j_message = (
                f"Neo4j has no record for class '{class_name}'."
            )

    return (
        get_defect_details(class_name, catalog),
        get_inventory_last_updated(catalog),
        "static_catalog_fallback",
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_defect(
    request: Request,
    file: UploadFile = File(...),
) -> PredictionResponse:
    """Validate, store, classify, and enrich one uploaded image."""
    settings = request.app.state.settings
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The trained model is not available yet.",
        )

    content_type = (file.content_type or "").lower()
    if content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only configured JPEG and PNG uploads are supported.",
        )

    contents = await file.read(settings.maximum_upload_size_bytes + 1)
    if len(contents) > settings.maximum_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.maximum_upload_size_mb} MB.",
        )
    original_filename = Path(file.filename or "").name
    if not validate_uploaded_image(
        original_filename,
        content_type,
        len(contents),
        settings.allowed_image_types,
        settings.maximum_upload_size_bytes,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image filename, extension, type, or size is invalid.",
        )

    try:
        image_sha256 = calculate_image_sha256(contents)
        stored_path = await run_in_threadpool(
            save_uploaded_image,
            contents,
            original_filename,
            settings.upload_directory,
        )
        image_size = int(request.app.state.model_metadata.get("image_size", 224))
        preprocessing_geometry = str(
            request.app.state.model_metadata.get("preprocessing", {}).get(
                "geometry", "resize_stretch"
            )
        )
        image_tensor = await run_in_threadpool(
            preprocess_image,
            stored_path,
            image_size,
            preprocessing_geometry,
        )
        prediction = await run_in_threadpool(
            predict_class,
            model,
            image_tensor,
            request.app.state.class_names,
            float(
                request.app.state.model_metadata.get(
                    "confidence_threshold", settings.confidence_threshold
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    feedback_class = await run_in_threadpool(
        find_feedback_class,
        image_sha256,
        settings.feedback_csv_path,
        set(request.app.state.class_names),
    )
    # Positive feedback can confirm the model's current result, but it has not
    # altered that result and should not be presented as a correction.
    feedback_applied = (
        feedback_class is not None
        and feedback_class != prediction.predicted_class
    )
    effective_class = feedback_class or prediction.predicted_class
    effective_requires_review = (
        prediction.requires_manual_review and not feedback_applied
    )

    if effective_requires_review:
        defect = None
        inventory_date = get_inventory_last_updated(request.app.state.catalog)
        knowledge_source = "static_catalog_fallback"
    else:
        defect, inventory_date, knowledge_source = await _load_defect_knowledge(
            request,
            effective_class,
        )
    if not effective_requires_review and defect is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The predicted class is missing from all knowledge sources.",
        )

    prediction_id = uuid4().hex
    if len(request.app.state.prediction_fingerprints) >= 1_000:
        oldest_prediction_id = next(
            iter(request.app.state.prediction_fingerprints)
        )
        request.app.state.prediction_fingerprints.pop(oldest_prediction_id)
    request.app.state.prediction_fingerprints[prediction_id] = image_sha256
    return PredictionResponse(
        prediction_id=prediction_id,
        filename=original_filename,
        image_sha256=image_sha256,
        predicted_class=effective_class,
        most_likely_class=prediction.most_likely_class,
        display_name="Unknown / Manual Review" if defect is None else defect.display_name,
        confidence=prediction.confidence,
        requires_manual_review=effective_requires_review,
        feedback_applied=feedback_applied,
        message=(
            (
                "Saved human correction applied for this exact image. "
                f"The model's uncorrected result was "
                f"'{prediction.most_likely_class}'."
            )
            if feedback_applied
            else (
                settings.unknown_message
                if prediction.requires_manual_review
                else prediction.message
            )
        ),
        repair_steps=[] if defect is None else defect.repair_steps,
        parts=[] if defect is None else defect.parts,
        inventory_last_updated=inventory_date,
        inventory_is_simulated=True,
        knowledge_source=knowledge_source,
        poc_disclaimer=POC_DISCLAIMER,
    )
