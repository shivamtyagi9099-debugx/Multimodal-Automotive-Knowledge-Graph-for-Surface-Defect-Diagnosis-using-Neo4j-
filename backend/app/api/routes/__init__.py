"""Route modules exposed by the proof-of-concept API."""

from backend.app.api.routes import feedback, health, prediction

__all__ = ["feedback", "health", "prediction"]
