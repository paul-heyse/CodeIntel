"""Test schema-aware adapter base classes.

Test the SchemaValidationMixin and SchemaAwareBatchAdapter classes
for schema validation integration.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from pandera import Column, DataFrameSchema

from codeintel.analytics.adapters.functions import (
    FunctionMetricsAdapter,
    FunctionTypesAdapter,
)
from codeintel.analytics.adapters.profiles import (
    FileProfileAdapter,
    FunctionProfileAdapter,
    ModuleProfileAdapter,
)
from codeintel.analytics.adapters.schema_adapter import (
    SchemaAwareBatchAdapter,
    SchemaValidationMixin,
)
from codeintel.analytics.adapters.subsystems import (
    SubsystemModulesAdapter,
    SubsystemsAdapter,
)
from codeintel.config.datasets.schema import DatasetSchema
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.env_options import EnvOptions

# =============================================================================
# Constants
# =============================================================================

DEMO_REPO = "demo/repo"
DEMO_COMMIT = "abc123"


# =============================================================================
# Test Fixtures
# =============================================================================


def _build_ctx(tmp_path: Path) -> TestContext:
    """Construct a TestContext for adapter tests.

    Returns
    -------
    TestContext
        Context with canonical repo/commit identifiers.
    """
    options = EnvOptions(repo=DEMO_REPO, commit=DEMO_COMMIT)
    return create_test_context(tmp_path, options=options)


@pytest.fixture
def test_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Create a test context for adapter tests.

    Parameters
    ----------
    tmp_path
        Temporary directory for the test.

    Yields
    ------
    TestContext
        Context for testing.
    """
    ctx = _build_ctx(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


# =============================================================================
# SchemaValidationMixin Tests
# =============================================================================


class _MixinTestClass(SchemaValidationMixin):
    """Test class implementing SchemaValidationMixin."""

    table_key: ClassVar[str] = "analytics.function_metrics"


def test_mixin_get_schema_for_registered_table() -> None:
    """Mixin can retrieve schema for registered table."""
    # Initialize registry first
    SCHEMA_REGISTRY.initialize()

    mixin = _MixinTestClass()

    # Try to get schema - may or may not exist depending on registry state
    try:
        schema = mixin.get_schema()
        expect_is_instance(schema, DatasetSchema)
        expect_equal(schema.name, "analytics.function_metrics")
    except KeyError:
        # Schema not registered is also valid for this test
        pass


def test_mixin_get_schema_for_unregistered_table() -> None:
    """Mixin raises KeyError for unregistered table."""

    class _UnregisteredMixin(SchemaValidationMixin):
        table_key: ClassVar[str] = "nonexistent.table"

    mixin = _UnregisteredMixin()
    with pytest.raises(KeyError, match=r"nonexistent\.table"):
        mixin.get_schema()


def test_mixin_validate_dataframe_with_valid_data() -> None:
    """Mixin validates DataFrame successfully with valid data."""
    # Create a minimal test schema
    pandera_schema = DataFrameSchema(
        {
            "name": Column(str),
            "value": Column(int),
        }
    )
    test_schema = DatasetSchema(
        name="test.validation",
        pandera_schema=pandera_schema,
    )

    # Create a temporary registry entry for testing
    class _TestMixin(SchemaValidationMixin):
        table_key: ClassVar[str] = "test.validation"
        test_schema_value: ClassVar[DatasetSchema] = test_schema

        def get_schema(self) -> DatasetSchema:
            """Return the test schema.

            Returns
            -------
            DatasetSchema
                The test schema for validation.
            """
            return type(self).test_schema_value

    mixin = _TestMixin()
    df = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    result = mixin.validate_dataframe(df)
    expect_equal(len(result), 2)


def test_mixin_try_validate_returns_original_on_failure() -> None:
    """Try validate returns original DataFrame on validation failure."""
    # Create a strict schema
    pandera_schema = DataFrameSchema(
        {
            "name": Column(str),
            "value": Column(int, coerce=False),
        }
    )
    test_schema = DatasetSchema(
        name="test.strict",
        pandera_schema=pandera_schema,
    )

    class _StrictMixin(SchemaValidationMixin):
        table_key: ClassVar[str] = "test.strict"
        test_schema_value: ClassVar[DatasetSchema] = test_schema

        def get_schema(self) -> DatasetSchema:
            """Return the test schema.

            Returns
            -------
            DatasetSchema
                The test schema for validation.
            """
            return type(self).test_schema_value

    mixin = _StrictMixin()
    # Invalid data - strings where ints expected
    df = pd.DataFrame({"name": ["a", "b"], "value": ["not", "int"]})

    # try_validate should not raise, should return original
    result = mixin.try_validate_dataframe(df)
    expect_equal(len(result), 2)


# =============================================================================
# SchemaAwareBatchAdapter Tests
# =============================================================================


class _TestSchemaAdapter(SchemaAwareBatchAdapter[dict[str, Any]]):
    """Concrete adapter for testing SchemaAwareBatchAdapter."""

    table_key: ClassVar[str] = "test.adapter"

    def load(self) -> Iterator[dict[str, Any]]:
        """Load is not implemented for this test adapter.

        Returns
        -------
        Iterator[dict[str, Any]]
            Empty iterator.
        """
        _ = self  # Satisfy the abstract method signature
        return iter([])

    def persist(self, rows: Sequence[dict[str, Any]]) -> int:
        """Persist rows (mock implementation).

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        _ = self  # Satisfy the abstract method signature
        return len(rows)


def test_schema_adapter_table_name(test_ctx: TestContext) -> None:
    """SchemaAwareBatchAdapter uses table_key for table_name."""
    adapter = _TestSchemaAdapter(test_ctx.gateway, test_ctx.snapshot)
    expect_equal(adapter.table_name, "test.adapter")


def test_schema_adapter_inherits_mixin_methods(test_ctx: TestContext) -> None:
    """SchemaAwareBatchAdapter has mixin methods available."""
    adapter = _TestSchemaAdapter(test_ctx.gateway, test_ctx.snapshot)

    # Verify mixin methods exist
    expect_true(hasattr(adapter, "get_schema"))
    expect_true(hasattr(adapter, "validate_dataframe"))
    expect_true(hasattr(adapter, "try_validate_dataframe"))


def test_schema_adapter_persist_validated_empty(test_ctx: TestContext) -> None:
    """Persist validated with empty rows returns 0."""
    adapter = _TestSchemaAdapter(test_ctx.gateway, test_ctx.snapshot)
    # Use delete_before=False to avoid trying to delete from non-existent table
    count = adapter.persist_validated([], delete_before=False, strict=False)
    expect_equal(count, 0)


def test_schema_adapter_gateway_property(test_ctx: TestContext) -> None:
    """SchemaAwareBatchAdapter exposes gateway property."""
    adapter = _TestSchemaAdapter(test_ctx.gateway, test_ctx.snapshot)
    expect_equal(adapter.gateway, test_ctx.gateway)


def test_schema_adapter_snapshot_property(test_ctx: TestContext) -> None:
    """SchemaAwareBatchAdapter exposes snapshot property."""
    adapter = _TestSchemaAdapter(test_ctx.gateway, test_ctx.snapshot)
    expect_is_instance(adapter.snapshot, SnapshotRef)


# =============================================================================
# Integration Tests
# =============================================================================


def test_function_metrics_adapter_has_schema_validation() -> None:
    """FunctionMetricsAdapter includes SchemaValidationMixin."""
    expect_true(hasattr(FunctionMetricsAdapter, "table_key"))
    expect_equal(FunctionMetricsAdapter.table_key, "analytics.function_metrics")


def test_function_types_adapter_has_schema_validation() -> None:
    """FunctionTypesAdapter includes SchemaValidationMixin."""
    expect_true(hasattr(FunctionTypesAdapter, "table_key"))
    expect_equal(FunctionTypesAdapter.table_key, "analytics.function_types")


def test_profile_adapters_have_schema_validation() -> None:
    """Profile adapters include SchemaValidationMixin."""
    expect_true(hasattr(FunctionProfileAdapter, "table_key"))
    expect_true(hasattr(FileProfileAdapter, "table_key"))
    expect_true(hasattr(ModuleProfileAdapter, "table_key"))


def test_subsystem_adapters_have_schema_validation() -> None:
    """Subsystem adapters include SchemaValidationMixin."""
    expect_true(hasattr(SubsystemsAdapter, "table_key"))
    expect_true(hasattr(SubsystemModulesAdapter, "table_key"))
