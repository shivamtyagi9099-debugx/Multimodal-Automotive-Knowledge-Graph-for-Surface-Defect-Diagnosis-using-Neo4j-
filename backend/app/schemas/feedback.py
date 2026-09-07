"""Schemas for locally stored prediction feedback."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeedbackRequest(BaseModel):
    """Represent one Y/N prediction-feedback submission."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(min_length=1)
    image_filename: str = Field(min_length=1)
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    predicted_class: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    corrected_class: str | None = None
    comment: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def require_correction_for_incorrect_prediction(self) -> "FeedbackRequest":
        """Require the label needed to correct a rejected prediction."""
        if not self.is_correct and self.corrected_class is None:
            raise ValueError(
                "corrected_class is required when the prediction is incorrect"
            )
        if self.is_correct and self.corrected_class is not None:
            raise ValueError(
                "corrected_class must be empty when the prediction is correct"
            )
        return self


class FeedbackResponse(BaseModel):
    """Acknowledge successful local feedback persistence."""

    prediction_id: str
    accepted: bool
    message: str
