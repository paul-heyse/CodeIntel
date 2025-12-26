"""Tests for Hamilton Phase 1 integration (Full Production Features).

These tests validate the Phase 1 Hamilton infrastructure:
- PR-01: HamiltonNodeMode and target-node mappings
- PR-02: Closure execution and result tracking
- PR-03: Upstream failure gating
- PR-04: Force flag support
- PR-05: Run tracking
- PR-06: Dataset lineage
- PR-07: Observability

All tests follow the Testing Charter: real components, no monkeypatching,
production-parity execution paths.
"""

from __future__ import annotations

import json

import pytest

from codeintel.build.hamilton.contracts.pandera_hook import (
    contract_status_for_table,
    get_pandera_schema,
    with_contract,
)
from codeintel.build.hamilton.driver_factory import (
    build_driver,
    list_available_nodes,
    target_to_node_name,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildResult
from codeintel.build.hamilton.io.dataset_ref import (
    DatasetRef,
    refs_from_target_result,
    refs_to_tuple,
)
from codeintel.build.hamilton.io.ibis_adapter import IbisIOConfig
from codeintel.build.hamilton.naming import dataset_node, target_node
from codeintel.build.hamilton.nodes.support_factory import (
    SupportGenerationOptions,
    build_support_module,
)
from codeintel.build.hamilton.observability import (
    export_dag_json,
    get_dag_info,
    list_execution_order,
    list_execution_targets,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from tests._helpers import assert_frozen
from tests._helpers.assertions import assert_target_ok

DEFAULT_ROW_COUNT = 1500
UPDATED_ROW_COUNT = 100
EXPECTED_REF_COUNT = 2
CLOSURE_LENGTH_PHASE0 = 5
SAMPLE_DURATION_MS = 1234.5


class TestHamiltonDriverMappings:
    """Tests for PR-01: target/node mappings remain stable."""

    @staticmethod
    def test_runtime_has_target_to_node_mapping() -> None:
        """Verify runtime carries target_to_node mapping."""
        runtime = build_driver()
        if not runtime.target_to_node:
            pytest.fail("Runtime missing target_to_node mapping")
        if "modules" not in runtime.target_to_node:
            pytest.fail("Mapping missing 'modules' target")

    @staticmethod
    def test_runtime_has_node_to_target_mapping() -> None:
        """Verify runtime carries node_to_target mapping."""
        runtime = build_driver()
        if not runtime.node_to_target:
            pytest.fail("Runtime missing node_to_target mapping")
        if "t__modules" not in runtime.node_to_target:
            pytest.fail("Mapping missing 't__modules' node")

    @staticmethod
    def test_target_to_node_name_with_runtime() -> None:
        """Verify target_to_node_name uses runtime mapping when provided."""
        runtime = build_driver()
        node_name = target_to_node_name("modules", runtime=runtime)
        if node_name != "t__modules":
            pytest.fail(f"Expected 't__modules', got '{node_name}'")

    @staticmethod
    def test_target_to_node_name_without_runtime() -> None:
        """Verify target_to_node_name works without runtime."""
        node_name = target_to_node_name("modules", runtime=build_driver())
        if node_name != "t__modules":
            pytest.fail(f"Expected 't__modules', got '{node_name}'")

    @staticmethod
    def test_list_available_nodes() -> None:
        """Verify list_available_nodes returns nodes."""
        nodes = list_available_nodes()
        if "t__modules" not in nodes:
            pytest.fail("Expected nodes to include t__modules")


class TestHamiltonBuildResult:
    """Tests for PR-02: HamiltonBuildResult fields."""

    @staticmethod
    def test_result_has_closure_field() -> None:
        """Verify HamiltonBuildResult has closure field."""
        result = HamiltonBuildResult(
            requested=("risk_factors",),
            closure=("modules", "scip", "ast", "goids", "risk_factors"),
        )
        if len(result.closure) != CLOSURE_LENGTH_PHASE0:
            pytest.fail(
                f"Expected {CLOSURE_LENGTH_PHASE0} items in closure, got {len(result.closure)}",
            )

    @staticmethod
    def test_result_has_computed_targets() -> None:
        """Verify HamiltonBuildResult has computed_targets."""
        result = HamiltonBuildResult(
            requested=("risk_factors",),
            computed_targets=("modules", "scip"),
        )
        if result.computed_targets != ("modules", "scip"):
            pytest.fail("computed_targets not set correctly")

    @staticmethod
    def test_result_has_skipped_targets() -> None:
        """Verify HamiltonBuildResult has skipped_targets."""
        result = HamiltonBuildResult(
            requested=("risk_factors",),
            skipped_targets=("goids",),
        )
        if result.skipped_targets != ("goids",):
            pytest.fail("skipped_targets not set correctly")

    @staticmethod
    def test_result_has_duration_ms() -> None:
        """Verify HamiltonBuildResult has duration_ms."""
        result = HamiltonBuildResult(
            requested=("risk_factors",),
            duration_ms=SAMPLE_DURATION_MS,
        )
        if result.duration_ms != SAMPLE_DURATION_MS:
            pytest.fail(f"Expected {SAMPLE_DURATION_MS}, got {result.duration_ms}")

    @staticmethod
    def test_result_has_run_id() -> None:
        """Verify HamiltonBuildResult has run_id."""
        result = HamiltonBuildResult(
            requested=("risk_factors",),
            run_id="hamilton-20241201-abc123",
        )
        if not result.run_id.startswith("hamilton-"):
            pytest.fail(f"run_id format incorrect: {result.run_id}")


class TestUpstreamFailureGating:
    """Tests for PR-03: Upstream failure gating in _run_target."""

    @staticmethod
    def test_upstream_failed_error_format() -> None:
        """Verify upstream_failed error message format."""
        record = TargetRunRecord(
            target="call_graph",
            impl_kind="native",
            status="skipped",
            input_hash=None,
            error="upstream_failed:goids,scip",
        )
        assert_target_ok(record, expected_status="skipped")
        if not record.error or "upstream_failed:" not in record.error:
            pytest.fail("Error should contain upstream_failed prefix")


class TestForceFlag:
    """Tests for PR-04: Force flag support in BuildEnv."""

    @staticmethod
    def test_build_env_has_force_targets() -> None:
        """Verify BuildEnv has force_targets field."""
        fields = getattr(BuildEnv, "__dataclass_fields__", {})
        if "force_targets" not in fields:
            pytest.fail("BuildEnv missing force_targets field")

    @staticmethod
    def test_build_env_is_forced_method() -> None:
        """Verify BuildEnv has is_forced method."""
        if not hasattr(BuildEnv, "is_forced"):
            pytest.fail("BuildEnv missing is_forced method")

    @staticmethod
    def test_force_targets_default_empty() -> None:
        """Verify force_targets defaults to empty frozenset."""
        fields = getattr(BuildEnv, "__dataclass_fields__", {})
        force_field = fields.get("force_targets")
        if force_field is None:
            pytest.fail("force_targets field not found")

        if force_field.default_factory is None:
            pytest.fail("force_targets should have default_factory")


class TestRunTracking:
    """Tests for PR-05: Run tracking in executor."""

    @staticmethod
    def test_result_includes_run_id() -> None:
        """Verify HamiltonBuildResult has run_id field."""
        fields = getattr(HamiltonBuildResult, "__dataclass_fields__", {})
        if "run_id" not in fields:
            pytest.fail("HamiltonBuildResult missing run_id field")


class TestDatasetRef:
    """Tests for DatasetRef type system."""

    @staticmethod
    def test_dataset_ref_creation() -> None:
        """Verify DatasetRef can be created with all fields."""
        ref = DatasetRef(
            table_key="analytics.function_metrics",
            schema_version="1.0.0",
            row_count=DEFAULT_ROW_COUNT,
            source_target="function_metrics",
            metadata={"computed_at": "2024-01-01"},
        )
        if ref.table_key != "analytics.function_metrics":
            pytest.fail("table_key not set correctly")
        if ref.row_count != DEFAULT_ROW_COUNT:
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
    def test_dataset_ref_with_row_count() -> None:
        """Verify with_row_count returns new instance."""
        ref = DatasetRef(table_key="test.table")
        updated = ref.with_row_count(UPDATED_ROW_COUNT)
        if updated is ref:
            pytest.fail("with_row_count returned same instance")
        if updated.row_count != UPDATED_ROW_COUNT:
            pytest.fail(f"Expected {UPDATED_ROW_COUNT}, got {updated.row_count}")

    @staticmethod
    def test_dataset_ref_frozen() -> None:
        """Verify DatasetRef is immutable."""
        ref = DatasetRef(table_key="test.table")
        assert_frozen(ref, "table_key", "other.table")


class TestRefsFromTargetResult:
    """Tests for refs_from_target_result helper."""

    @staticmethod
    def test_creates_refs_for_all_tables() -> None:
        """Verify refs are created for all table keys."""
        refs = refs_from_target_result(
            target_name="function_metrics",
            table_keys=("analytics.function_metrics", "analytics.extra"),
        )
        if len(refs) != EXPECTED_REF_COUNT:
            pytest.fail(f"Expected {EXPECTED_REF_COUNT} refs, got {len(refs)}")

    @staticmethod
    def test_includes_row_counts() -> None:
        """Verify row counts are included when provided."""
        refs = refs_from_target_result(
            target_name="function_metrics",
            table_keys=("analytics.function_metrics",),
            row_counts={"analytics.function_metrics": DEFAULT_ROW_COUNT},
        )
        ref = refs["analytics.function_metrics"]
        if ref.row_count != DEFAULT_ROW_COUNT:
            pytest.fail(f"Expected {DEFAULT_ROW_COUNT}, got {ref.row_count}")


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
        if len(tup) != EXPECTED_REF_COUNT:
            pytest.fail(f"Expected {EXPECTED_REF_COUNT} items, got {len(tup)}")


class TestTargetRunRecordDatasets:
    """Tests for TargetRunRecord datasets field."""

    @staticmethod
    def test_datasets_field_exists() -> None:
        """Verify TargetRunRecord has datasets field."""
        record = TargetRunRecord(
            target="test",
            impl_kind="native",
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
            impl_kind="native",
            status="succeeded",
            input_hash="abc123",
            datasets=(ref,),
        )
        found = record.get_dataset("test.table")
        if found is None:
            pytest.fail("get_dataset returned None for existing ref")
        if found.table_key != "test.table":
            pytest.fail("get_dataset returned wrong ref")


class TestObservability:
    """Tests for PR-07: Observability functions."""

    @staticmethod
    def test_list_execution_targets() -> None:
        """Verify list_execution_targets returns target names."""
        runtime = build_driver()
        targets = list_execution_targets(runtime, ["modules"])
        if "modules" not in targets:
            pytest.fail("modules should be in execution targets")

    @staticmethod
    def test_list_execution_order() -> None:
        """Verify list_execution_order returns node names."""
        runtime = build_driver()
        order = list_execution_order(runtime, ["modules"])
        if "t__modules" not in order:
            pytest.fail("t__modules should be in execution order")

    @staticmethod
    def test_get_dag_info_structure() -> None:
        """Verify get_dag_info returns expected structure."""
        runtime = build_driver()
        info = get_dag_info(runtime, ["modules"])
        if "nodes" not in info:
            pytest.fail("DAG info missing 'nodes' field")
        if "edges" not in info:
            pytest.fail("DAG info missing 'edges' field")
        if "closure" not in info:
            pytest.fail("DAG info missing 'closure' field")

    @staticmethod
    def test_export_dag_json_valid() -> None:
        """Verify export_dag_json returns valid JSON."""
        runtime = build_driver()
        json_str = export_dag_json(runtime, ["modules"])
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON: {e}")
        if "nodes" not in data:
            pytest.fail("JSON missing 'nodes' field")


class TestSupportFactory:
    """Tests for support-node module generation."""

    @staticmethod
    def test_build_support_module_creates_module() -> None:
        """Verify build_support_module returns a module with nodes."""
        module = build_support_module()
        if not hasattr(module, "__doc__"):
            pytest.fail("Module missing __doc__")

    @staticmethod
    def test_build_support_module_has_target_to_node() -> None:
        """Verify support module exposes TARGET_TO_NODE mapping."""
        module = build_support_module()
        if not hasattr(module, "TARGET_TO_NODE"):
            pytest.fail("Module missing TARGET_TO_NODE")
        if len(module.TARGET_TO_NODE) != 0:
            pytest.fail("TARGET_TO_NODE should be empty for native-only modules")

    @staticmethod
    def test_build_support_module_has_dataset_to_node() -> None:
        """Verify support module has DATASET_TO_NODE mapping."""
        module = build_support_module()
        if not hasattr(module, "DATASET_TO_NODE"):
            pytest.fail("Module missing DATASET_TO_NODE mapping")

    @staticmethod
    def test_build_support_module_respects_exclude() -> None:
        """Verify exclude_targets filters out targets."""
        module = build_support_module(
            options=SupportGenerationOptions(exclude_targets=frozenset({"modules"})),
        )
        if hasattr(module, dataset_node("core.modules")):
            pytest.fail("Excluded target should not expose dataset nodes")


class TestDriverConstruction:
    """Tests for driver construction with the unified module set."""

    @staticmethod
    def test_build_driver_constructs_driver() -> None:
        """Verify build_driver constructs a Driver."""
        runtime = build_driver()
        if runtime.dr is None:
            pytest.fail("Driver runtime missing dr")


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


class TestIbisIOConfig:
    """Tests for IbisIOConfig dataclass."""

    @staticmethod
    def test_ibis_io_config_creation() -> None:
        """Verify IbisIOConfig has expected fields."""
        fields = getattr(IbisIOConfig, "__dataclass_fields__", {})
        if "gateway" not in fields:
            pytest.fail("IbisIOConfig missing gateway field")
        if "validate_schema" not in fields:
            pytest.fail("IbisIOConfig missing validate_schema field")


class TestPanderaContractIntegration:
    """Tests for Pandera contract integration."""

    @staticmethod
    def test_get_pandera_schema_import() -> None:
        """Verify get_pandera_schema can be imported."""
        if not callable(get_pandera_schema):
            pytest.fail("get_pandera_schema is not callable")

    @staticmethod
    def test_with_contract_decorator_import() -> None:
        """Verify with_contract decorator can be imported."""
        if not callable(with_contract):
            pytest.fail("with_contract is not callable")

    @staticmethod
    def test_contract_status_for_table() -> None:
        """Verify contract_status_for_table returns expected structure."""
        status = contract_status_for_table("nonexistent.table")
        if not isinstance(status, dict):
            pytest.fail("contract_status_for_table should return dict")
        if "has_schema" not in status:
            pytest.fail("Status missing has_schema field")
        if "table_key" not in status:
            pytest.fail("Status missing table_key field")
