"""Service functions for inference, catalog, Neo4j, feedback, and images."""

from backend.app.services.catalog_service import (
    get_defect_details,
    get_inventory_last_updated,
    load_catalog,
)
from backend.app.services.feedback_service import append_feedback
from backend.app.services.image_service import (
    save_uploaded_image,
    validate_uploaded_image,
)
from backend.app.services.inference_service import (
    load_model,
    load_model_bundle,
    predict_class,
    preprocess_image,
)
from backend.app.services.neo4j_service import (
    DefectKnowledgeService,
    Neo4jConfigurationError,
    Neo4jService,
    Neo4jServiceError,
)

__all__ = [
    "append_feedback",
    "get_defect_details",
    "get_inventory_last_updated",
    "load_catalog",
    "load_model",
    "load_model_bundle",
    "DefectKnowledgeService",
    "Neo4jConfigurationError",
    "Neo4jService",
    "Neo4jServiceError",
    "predict_class",
    "preprocess_image",
    "save_uploaded_image",
    "validate_uploaded_image",
]
