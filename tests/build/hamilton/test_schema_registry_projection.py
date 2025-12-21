"""Tests for schema registry projection from unified providers."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
from codeintel.build.schemas import get_schema_provider


def test_schema_registry_projects_table_schema() -> None:
    """Registry entries should project from the canonical schema provider."""
    table_key = "core.modules"
    schema = SCHEMA_REGISTRY.require(table_key)
    provider_schema = get_schema_provider().require_table_schema(table_key)

    if schema.ddl_schema != provider_schema:
        pytest.fail("Schema registry DDL schema should match the canonical provider schema.")

    if schema.column_names() != tuple(provider_schema.column_names()):
        pytest.fail("Schema registry Pandera columns should match TableSchema ordering.")
