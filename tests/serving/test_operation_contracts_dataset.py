"""Validation tests for operation contract dataset reflection."""

from __future__ import annotations

import pandas as pd
import pytest

from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY
from codeintel.serving.contracts.operation_contract_reflection import (
    ComponentSpec,
    build_operation_contract_dataframe,
    validate_operation_contracts,
)
from codeintel.serving.contracts.operation_contracts_dataset import (
    OPERATION_CONTRACT_TABLE_KEY,
)
from codeintel.serving.mcp.backend import DuckDBBackend, HttpBackend
from codeintel.serving.services.query_service import HttpQueryService, LocalQueryService
from codeintel.serving.types import QueryBackendProtocol, QueryServiceProtocol


@pytest.fixture
def operation_components() -> list[ComponentSpec]:
    """Components to reflect for contract validation."""
    return [
        ComponentSpec(
            component=QueryServiceProtocol,
            transport="protocol_service",
        ),
        ComponentSpec(
            component=QueryBackendProtocol,
            transport="protocol_backend",
        ),
        ComponentSpec(
            component=LocalQueryService,
            transport="service",
        ),
        ComponentSpec(
            component=HttpQueryService,
            transport="service",
        ),
        ComponentSpec(
            component=DuckDBBackend,
            transport="backend",
        ),
        ComponentSpec(
            component=HttpBackend,
            transport="backend",
        ),
    ]


def test_operation_contract_schema_registered() -> None:
    """Dataset schema should be registered in the global registry."""
    schema = SCHEMA_REGISTRY.require(OPERATION_CONTRACT_TABLE_KEY)
    assert schema.name == OPERATION_CONTRACT_TABLE_KEY
    assert "component" in schema.column_names()


def test_operation_contract_reflection_validates(
    operation_components: list[ComponentSpec],
) -> None:
    """Reflected contracts validate against the DatasetSchema."""
    df = build_operation_contract_dataframe(operation_components)
    assert not df.empty

    validated = validate_operation_contracts(df)
    assert isinstance(validated, pd.DataFrame)
    assert not validated.empty
