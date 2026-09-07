"""Endpoint for append-only local prediction feedback."""

from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.feedback import FeedbackRequest, FeedbackResponse
from backend.app.services.feedback_service import append_feedback

router = APIRouter(prefix="/api/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: Request,
    payload: FeedbackRequest,
) -> FeedbackResponse:
    """Validate class references and append one local feedback record."""
    class_names = set(request.app.state.class_names)
    if payload.predicted_class not in class_names | {"unknown"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="predicted_class is not supported by this project.",
        )
    if payload.corrected_class is not None and payload.corrected_class not in class_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="corrected_class is not supported by this project.",
        )

    image_sha256 = request.app.state.prediction_fingerprints.pop(
        payload.prediction_id,
        payload.image_sha256,
    )
    await run_in_threadpool(
        append_feedback,
        payload,
        request.app.state.settings.feedback_csv_path,
        image_sha256,
    )
    return FeedbackResponse(
        prediction_id=payload.prediction_id,
        accepted=True,
        message=(
            "Feedback saved. It will be applied when this exact image is "
            "analysed again."
            if image_sha256
            else "Feedback recorded locally for future evaluation."
        ),
    )
