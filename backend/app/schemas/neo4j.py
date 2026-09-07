"""Typed settings and status records for the Neo4j knowledge graph."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class Neo4jConnectionSettings(BaseModel):
    """Validated connection information without exposing the password."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: SecretStr = Field(min_length=8)
    database: str = Field(default="neo4j", min_length=1)

    @field_validator("uri")
    @classmethod
    def validate_uri_scheme(cls, value: str) -> str:
        """Accept only Neo4j Bolt-compatible URI schemes."""
        normalized = value.strip()
        allowed_schemes = ("bolt://", "bolt+s://", "bolt+ssc://", "neo4j://", "neo4j+s://", "neo4j+ssc://")
        if not normalized.startswith(allowed_schemes):
            raise ValueError("Neo4j URI must use a bolt or neo4j scheme")
        return normalized

    @field_validator("username", "database")
    @classmethod
    def strip_non_empty_values(cls, value: str) -> str:
        """Remove accidental surrounding whitespace from identifiers."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Neo4j identifiers must not be blank")
        return normalized


class Neo4jHealthStatus(BaseModel):
    """Describe whether the configured Neo4j database is reachable."""

    model_config = ConfigDict(extra="forbid")

    connected: bool
    database: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    message: str = Field(min_length=1)


class GraphSeedSummary(BaseModel):
    """Report the number of catalog records submitted to the graph."""

    model_config = ConfigDict(extra="forbid")

    object_types: int = Field(ge=0)
    defects: int = Field(ge=0)
    repair_steps: int = Field(ge=0)
    parts: int = Field(ge=0)
    inventory_records: int = Field(ge=0)
