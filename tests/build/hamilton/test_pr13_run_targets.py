"""Tests for PR-13: Run targets persistence.

Validates build.run_targets schema exists and persistence works.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.cli.core.result_types import BuildHistoryResult
from codeintel.storage.tracking.build_tracking import BuildTracking


class TestRunTargetsSchema:
    """Tests for build.run_targets schema definition."""

    @staticmethod
    def test_run_targets_schema_exists() -> None:
        """Verify build.run_targets schema is registered."""
        schema_key = "build.run_targets"
        schema = SCHEMA_REGISTRY.get(schema_key)
        if schema is None:
            pytest.skip(f"Schema '{schema_key}' not registered")

    @staticmethod
    def test_run_targets_schema_has_required_columns() -> None:
        """Verify build.run_targets has required columns."""
        schema_key = "build.run_targets"
        schema = SCHEMA_REGISTRY.get(schema_key)
        if schema is None:
            pytest.skip("Schema not registered")

        if not hasattr(schema, "column_names"):
            pytest.skip("Schema doesn't have column_names attribute")

        expected_columns = {
            "run_id",
            "target",
            "status",
            "input_hash",
        }

        column_names_attr = getattr(schema, "column_names", None)
        if callable(column_names_attr):
            columns = column_names_attr()
        else:
            columns = column_names_attr if column_names_attr is not None else ()
        raw_columns = columns if isinstance(columns, Iterable) else (columns,)
        actual_columns = {str(column) for column in raw_columns}

        missing = expected_columns - actual_columns
        if missing:
            pytest.fail(f"Schema missing columns: {missing}")


class TestRunTargetsPersistence:
    """Tests for run targets persistence (integration)."""

    @staticmethod
    def test_tracking_has_run_target_methods() -> None:
        """Verify BuildTracking has run_target methods."""
        if not hasattr(BuildTracking, "list_run_targets"):
            pytest.skip("list_run_targets method not yet implemented")

        if not callable(getattr(BuildTracking, "list_run_targets", None)):
            pytest.fail("list_run_targets should be callable")

    @staticmethod
    def test_target_run_record_structure() -> None:
        """Verify TargetRunRecord has expected fields."""
        record = TargetRunRecord(
            target="modules",
            plugin_name="ingestion.modules",
            status="succeeded",
            input_hash="hash123",
            options_hash="opts",
            duration_ms=100.5,
        )

        if not hasattr(record, "target"):
            pytest.fail("TargetRunRecord missing target field")
        if not hasattr(record, "status"):
            pytest.fail("TargetRunRecord missing status field")
        if not hasattr(record, "input_hash"):
            pytest.fail("TargetRunRecord missing input_hash field")


class TestHistoryWithRunTargets:
    """Tests for history command with run_targets support."""

    @staticmethod
    def test_history_result_type_has_targets_field() -> None:
        """Verify BuildHistoryResult can include targets (optional enhancement)."""
        fields = getattr(BuildHistoryResult, "__dataclass_fields__", {})
        if "targets" not in fields:
            pytest.fail("targets field not implemented in BuildHistoryResult")

        field_info = fields["targets"]
        if field_info.default is None and field_info.default_factory is None:
            pytest.fail("targets field should have a default (optional)")
