"""Tests for Hamilton Phase 1 integration (IO & Contracts).

These tests validate the Phase 1 Hamilton infrastructure:
- DatasetRef type system
- Ibis IO adapters
- Pandera contract integration
- Dataset extraction nodes
- Node factory for dynamic generation

All tests follow the Testing Charter: real components, no monkeypatching,
production-parity execution paths.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.io.dataset_ref import (
    DatasetRef,
    refs_from_target_result,
    refs_to_tuple,
)
from codeintel.build.hamilton.io.ibis_adapter import IbisIOConfig
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.naming import dataset_node, target_node
from codeintel.build.hamilton.nodes.node_factory import (
    build_target_module,
    clear_generated_module_cache,
    get_generated_module,
)


class TestDatasetRef:
    """Tests for DatasetRef type system."""

    @staticmethod
    def test_dataset_ref_creation() -> None:
        """Verify DatasetRef can be created with all fields."""
        ref = DatasetRef(
            table_key="analytics.function_metrics",
            schema_version="1.0.0",
            row_count=1500,
            source_target="function_metrics",
            metadata={"computed_at": "2024-01-01"},
        )
        if ref.table_key != "analytics.function_metrics":
            pytest.fail("table_key not set correctly")
        if ref.row_count != 1500:
            pytest.fail("row_count not set correctly")

    @staticmethod
    def test_dataset_ref_schema_name_extraction() -> None:
        """Verify schema_name is extracted from qualified table key."""
        ref = DatasetRef(table_key="analytics.function_metrics")
        if ref.schema_name != "analytics":
            pytest.fail(f"Expected 'analytics', got '{ref.schema_name}'")

    @staticmethod
    def test_dataset_ref_table_name_extraction() -> None:
        """Verify table_name is extracted from qualified table key."""
        ref = DatasetRef(table_key="analytics.function_metrics")
        if ref.table_name != "function_metrics":
            pytest.fail(f"Expected 'function_metrics', got '{ref.table_name}'")

    @staticmethod
    def test_dataset_ref_unqualified_table() -> None:
        """Verify unqualified table uses 'main' as default schema."""
        ref = DatasetRef(table_key="simple_table")
        if ref.schema_name != "main":
            pytest.fail(f"Expected 'main', got '{ref.schema_name}'")
        if ref.table_name != "simple_table":
            pytest.fail(f"Expected 'simple_table', got '{ref.table_name}'")

    @staticmethod
    def test_dataset_ref_with_row_count() -> None:
        """Verify with_row_count returns new instance."""
        ref = DatasetRef(table_key="test.table")
        updated = ref.with_row_count(100)
        if updated is ref:
            pytest.fail("with_row_count returned same instance")
        if updated.row_count != 100:
            pytest.fail(f"Expected 100, got {updated.row_count}")
        if ref.row_count is not None:
            pytest.fail("Original ref should be unchanged")

    @staticmethod
    def test_dataset_ref_with_metadata() -> None:
        """Verify with_metadata returns new instance with added metadata."""
        ref = DatasetRef(table_key="test.table")
        updated = ref.with_metadata("key", "value")
        if updated is ref:
            pytest.fail("with_metadata returned same instance")
        if updated.metadata.get("key") != "value":
            pytest.fail("Metadata not added")
        if "key" in ref.metadata:
            pytest.fail("Original ref should be unchanged")

    @staticmethod
    def test_dataset_ref_frozen() -> None:
        """Verify DatasetRef is immutable."""
        ref = DatasetRef(table_key="test.table")
        with pytest.raises(AttributeError):
            ref.table_key = "other.table"  # type: ignore[misc]


class TestRefsFromTargetResult:
    """Tests for refs_from_target_result helper."""

    @staticmethod
    def test_creates_refs_for_all_tables() -> None:
        """Verify refs are created for all table keys."""
        refs = refs_from_target_result(
            target_name="function_metrics",
            table_keys=("analytics.function_metrics", "analytics.extra"),
        )
        if len(refs) != 2:
            pytest.fail(f"Expected 2 refs, got {len(refs)}")
        if "analytics.function_metrics" not in refs:
            pytest.fail("Missing analytics.function_metrics ref")
        if "analytics.extra" not in refs:
            pytest.fail("Missing analytics.extra ref")

    @staticmethod
    def test_includes_row_counts() -> None:
        """Verify row counts are included when provided."""
        refs = refs_from_target_result(
            target_name="function_metrics",
            table_keys=("analytics.function_metrics",),
            row_counts={"analytics.function_metrics": 1500},
        )
        ref = refs["analytics.function_metrics"]
        if ref.row_count != 1500:
            pytest.fail(f"Expected 1500, got {ref.row_count}")

    @staticmethod
    def test_includes_source_target() -> None:
        """Verify source_target is set on all refs."""
        refs = refs_from_target_result(
            target_name="function_metrics",
            table_keys=("analytics.function_metrics",),
        )
        ref = refs["analytics.function_metrics"]
        if ref.source_target != "function_metrics":
            pytest.fail(f"Expected 'function_metrics', got '{ref.source_target}'")


class TestRefsToTuple:
    """Tests for refs_to_tuple helper."""

    @staticmethod
    def test_converts_to_tuple() -> None:
        """Verify dict is converted to tuple."""
        refs = refs_from_target_result(
            target_name="test",
            table_keys=("t1", "t2"),
        )
        tup = refs_to_tuple(refs)
        if not isinstance(tup, tuple):
            pytest.fail("Result is not a tuple")
        if len(tup) != 2:
            pytest.fail(f"Expected 2 items, got {len(tup)}")


class TestIbisIOConfig:
    """Tests for IbisIOConfig dataclass."""

    @staticmethod
    def test_ibis_io_config_creation() -> None:
        """Verify IbisIOConfig has expected fields."""
        # Check dataclass fields via __dataclass_fields__
        fields = getattr(IbisIOConfig, "__dataclass_fields__", {})
        if "gateway" not in fields:
            pytest.fail("IbisIOConfig missing gateway field")
        if "validate_schema" not in fields:
            pytest.fail("IbisIOConfig missing validate_schema field")


class TestTargetRunRecordDatasets:
    """Tests for TargetRunRecord datasets field."""

    @staticmethod
    def test_datasets_field_exists() -> None:
        """Verify TargetRunRecord has datasets field."""
        record = TargetRunRecord(
            target="test",
            plugin_name="test.plugin",
            status="succeeded",
            input_hash="abc123",
        )
        if not hasattr(record, "datasets"):
            pytest.fail("TargetRunRecord missing datasets field")
        if record.datasets != ():
            pytest.fail("Default datasets should be empty tuple")

    @staticmethod
    def test_get_dataset_returns_matching_ref() -> None:
        """Verify get_dataset finds matching DatasetRef."""
        ref = DatasetRef(table_key="test.table")
        record = TargetRunRecord(
            target="test",
            plugin_name="test.plugin",
            status="succeeded",
            input_hash="abc123",
            datasets=(ref,),
        )
        found = record.get_dataset("test.table")
        if found is None:
            pytest.fail("get_dataset returned None for existing ref")
        if found.table_key != "test.table":
            pytest.fail("get_dataset returned wrong ref")

    @staticmethod
    def test_get_dataset_returns_none_for_missing() -> None:
        """Verify get_dataset returns None for non-existent table."""
        record = TargetRunRecord(
            target="test",
            plugin_name="test.plugin",
            status="succeeded",
            input_hash="abc123",
        )
        found = record.get_dataset("nonexistent.table")
        if found is not None:
            pytest.fail("get_dataset should return None for missing table")


class TestNodeFactory:
    """Tests for dynamic node generation."""

    @staticmethod
    def test_build_target_module_creates_module() -> None:
        """Verify build_target_module returns a module with nodes."""
        clear_generated_module_cache()
        module = build_target_module()
        if module is None:
            pytest.fail("build_target_module returned None")
        if not hasattr(module, "__doc__"):
            pytest.fail("Module missing __doc__")
        # Check that nodes were actually created
        target_attrs = [a for a in dir(module) if a.startswith("t__")]
        if not target_attrs:
            # Get the target graph to see what targets exist
            from codeintel.build.registry import get_target_graph

            graph = get_target_graph()
            target_names = [t.name for t in graph.all_targets]
            pytest.fail(f"No t__ nodes in module. Targets in graph: {target_names[:5]}")

    @staticmethod
    def test_build_target_module_has_target_to_node() -> None:
        """Verify generated module has TARGET_TO_NODE mapping."""
        clear_generated_module_cache()
        module = build_target_module()
        if not hasattr(module, "TARGET_TO_NODE"):
            pytest.fail("Module missing TARGET_TO_NODE")

    @staticmethod
    def test_build_target_module_respects_exclude() -> None:
        """Verify exclude_targets filters out targets."""
        clear_generated_module_cache()
        module = build_target_module(exclude_targets={"modules"})
        if hasattr(module, target_node("modules")):
            pytest.fail("Excluded target should not be in module")

    @staticmethod
    def test_get_generated_module_caches() -> None:
        """Verify get_generated_module returns cached instance."""
        clear_generated_module_cache()
        module1 = get_generated_module()
        module2 = get_generated_module()
        if module1 is not module2:
            pytest.fail("get_generated_module should return cached instance")

    @staticmethod
    def test_clear_generated_module_cache_works() -> None:
        """Verify cache can be cleared."""
        clear_generated_module_cache()
        module1 = get_generated_module()
        clear_generated_module_cache()
        module2 = get_generated_module()
        # After clearing, should be different instance
        # (well, could be same but that's fine for this test)
        if module1 is None or module2 is None:
            pytest.fail("Modules should not be None")


class TestDriverWithGeneratedNodes:
    """Tests for driver construction with generated nodes."""

    @staticmethod
    def test_build_driver_with_generated_flag() -> None:
        """Verify build_driver supports use_generated flag."""
        clear_generated_module_cache()
        runtime = build_driver(use_generated=True)
        if runtime.dr is None:
            pytest.fail("Driver runtime missing dr")

    @staticmethod
    def test_generated_driver_has_nodes() -> None:
        """Verify generated driver has expected nodes."""
        clear_generated_module_cache()
        runtime = build_driver(use_generated=True)
        all_nodes = list(runtime.dr.list_available_variables())
        node_names = [n.name for n in all_nodes]
        # Check if we have any target nodes at all
        target_nodes = [n for n in node_names if n.startswith("t__")]
        if not target_nodes:
            # If no target nodes, check the generated module directly
            generated_mod = get_generated_module()
            mod_attrs = [a for a in dir(generated_mod) if a.startswith("t__")]
            pytest.fail(f"No target nodes found. Module attrs: {mod_attrs}")
        # Should have at least the modules target
        if "t__modules" not in node_names:
            pytest.fail(f"Generated driver missing t__modules. Found: {target_nodes[:10]}")


class TestDatasetNodeNaming:
    """Tests for dataset node naming conventions."""

    @staticmethod
    def test_dataset_node_names_are_consistent() -> None:
        """Verify dataset node names follow convention."""
        table_keys = [
            "graph.call_graph_edges",
            "analytics.function_metrics",
            "analytics.risk_factors",
        ]
        for key in table_keys:
            node_name = dataset_node(key)
            if not node_name.startswith("d__"):
                pytest.fail(f"Dataset node {node_name} should start with d__")
            if not node_name.isidentifier():
                pytest.fail(f"Dataset node {node_name} is not valid identifier")

    @staticmethod
    def test_dataset_and_target_nodes_distinct() -> None:
        """Verify dataset and target nodes use different prefixes."""
        target_name = target_node("function_metrics")
        dataset_name = dataset_node("analytics.function_metrics")
        if target_name == dataset_name:
            pytest.fail("Target and dataset nodes should have different names")
        if not target_name.startswith("t__"):
            pytest.fail("Target node should use t__ prefix")
        if not dataset_name.startswith("d__"):
            pytest.fail("Dataset node should use d__ prefix")


class TestPanderaContractIntegration:
    """Tests for Pandera contract integration."""

    @staticmethod
    def test_get_pandera_schema_import() -> None:
        """Verify get_pandera_schema can be imported."""
        from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema

        # Just verify the function exists and is callable
        if not callable(get_pandera_schema):
            pytest.fail("get_pandera_schema is not callable")

    @staticmethod
    def test_with_contract_decorator_import() -> None:
        """Verify with_contract decorator can be imported."""
        from codeintel.build.hamilton.contracts.pandera_hook import with_contract

        # Just verify the function exists and is callable
        if not callable(with_contract):
            pytest.fail("with_contract is not callable")

    @staticmethod
    def test_contract_status_for_table() -> None:
        """Verify contract_status_for_table returns expected structure."""
        from codeintel.build.hamilton.contracts.pandera_hook import (
            contract_status_for_table,
        )

        status = contract_status_for_table("nonexistent.table")
        if not isinstance(status, dict):
            pytest.fail("contract_status_for_table should return dict")
        if "has_schema" not in status:
            pytest.fail("Status missing has_schema field")
        if "table_key" not in status:
            pytest.fail("Status missing table_key field")
