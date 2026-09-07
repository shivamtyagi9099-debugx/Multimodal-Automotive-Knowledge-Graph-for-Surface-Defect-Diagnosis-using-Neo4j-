"""Health-check endpoint for local PoC readiness."""

from fastapi import APIRouter, Request

from backend.app.schemas.prediction import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Report inference readiness and refresh Neo4j connectivity status."""
    model_loaded = request.app.state.model is not None
    catalog_loaded = request.app.state.catalog is not None
    neo4j_service = request.app.state.neo4j_service
    neo4j_connected = False
    if neo4j_service is not None:
        graph_status = neo4j_service.verify_connectivity()
        neo4j_connected = graph_status.connected
        request.app.state.neo4j_connected = graph_status.connected
        request.app.state.neo4j_message = graph_status.message

    knowledge_source = (
        "neo4j" if neo4j_connected else "static_catalog_fallback"
    )
    return HealthResponse(
        status=(
            "healthy"
            if model_loaded and catalog_loaded and neo4j_connected
            else "degraded"
        ),
        model_loaded=model_loaded,
        catalog_loaded=catalog_loaded,
        neo4j_configured=request.app.state.neo4j_configured,
        neo4j_connected=neo4j_connected,
        knowledge_source=knowledge_source,
        neo4j_message=request.app.state.neo4j_message,
    )
