"""Schemas for the local static repair and simulated inventory catalogue."""

from pydantic import BaseModel, ConfigDict, Field


class PartRecord(BaseModel):
    """Represent one simulated replacement-part inventory record."""

    model_config = ConfigDict(extra="forbid")

    part_id: str = Field(min_length=1)
    part_name: str = Field(min_length=1)
    mock_price: float = Field(ge=0.0)
    currency: str = Field(min_length=3, max_length=3)
    fake_stock_quantity: int = Field(ge=0)
    availability_status: str = Field(min_length=1)


class DefectCatalogEntry(BaseModel):
    """Represent one class and its manually verified catalogue content."""

    model_config = ConfigDict(extra="forbid")

    class_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    repair_steps: list[str]
    parts: list[PartRecord]


class InventoryCatalog(BaseModel):
    """Represent the entire local catalogue and transparency metadata."""

    model_config = ConfigDict(extra="forbid")

    object_type: str = Field(min_length=1)
    inventory_last_updated: str = Field(min_length=1)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    unknown_message: str = Field(min_length=1)
    defects: list[DefectCatalogEntry] = Field(min_length=1)
