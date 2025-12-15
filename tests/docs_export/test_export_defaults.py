"""Tests for export validation defaults derived from the dataset contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.build.schemas import get_contract_provider
from codeintel.export import default_validation_schemas


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_default_validation_schemas_match_dataset_contract() -> None:
    """Default validation schemas should mirror the dataset contract mapping."""
    json_schema_mapping = get_contract_provider().json_schema_by_dataset_name
    expected = sorted(json_schema_mapping.keys())
    dynamic = sorted(default_validation_schemas())
    _require(
        condition=dynamic == expected,
        message=f"default_validation_schemas mismatch: {dynamic} != {expected}",
    )


def test_schema_files_match_contract() -> None:
    """Ensure JSON Schema filenames align with the dataset contract."""
    schema_dir = Path("src/codeintel/config/schemas/export")
    stems = sorted(path.stem for path in schema_dir.glob("*.json") if path.stem != "base")
    json_schema_mapping = get_contract_provider().json_schema_by_dataset_name
    expected = sorted(set(json_schema_mapping.values()))
    _require(
        condition=stems == expected,
        message=f"Schema files do not match contract: {stems} != {expected}",
    )
