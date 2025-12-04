"""Consolidated tests for ingestion plugin infrastructure and registry.

This module brings together all plugin-related tests:
- Plugin protocol, metadata, and result types
- Plugin registry registration and lookup
- Plan creation and dependency resolution
- Custom plugin execution scenarios

Uses real plugins from DEFAULT_INGEST_PLUGINS where appropriate,
and test doubles that implement the full protocol for edge case testing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.core.base import (
    BaseIngestPlugin,
    TableWriterIngestPlugin,
)
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.plugins import (
    DEFAULT_INGEST_PLUGINS,
    IngestRuntimeScratch,
    get_ingest_registry,
    plan_ingest_plugins,
)
from codeintel.ingestion.plugins.protocol import (
    IngestIsolationKind,
    IngestPluginMetadata,
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginResult,
    IngestPluginSkip,
    IngestResourceHints,
    IngestSeverity,
    IngestStage,
    is_ingest_plugin,
)
from codeintel.ingestion.plugins.registry import (
    IngestPluginRegistry,
    PlanOptions,
)
from codeintel.ingestion.resources.registry import ResourceRegistry
from codeintel.ingestion.utilities.scanning import (
    default_code_profile,
    default_config_profile,
)
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway

# =============================================================================
# Test Constants
# =============================================================================

RUNTIME_MS_5000 = 5000
MEMORY_MB_512 = 512
RUNTIME_MS_3000 = 3000
ROW_COUNT_42 = 42
ROW_COUNT_5 = 5
ROW_COUNT_3 = 3
ROW_COUNT_10 = 10
TEST_ARTIFACT_PATH = Path("/opt/test/index.scip")


# =============================================================================
# Test Plugins
# =============================================================================


@dataclass
class SamplePlugin(BaseIngestPlugin):
    """Sample plugin for testing basic functionality."""

    plugin_name: ClassVar[str] = "sample_plugin"
    plugin_description: ClassVar[str] = "A sample test plugin"
    plugin_stage: ClassVar[IngestStage] = "parse"

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Return test row counts.

        Returns
        -------
        Mapping[str, int] | None
            Test row counts.
        """
        _ = self, ctx
        return {"core.test": ROW_COUNT_10}


@dataclass
class FailingPlugin(BaseIngestPlugin):
    """Plugin that always fails."""

    plugin_name: ClassVar[str] = "failing_plugin"
    plugin_description: ClassVar[str] = "A plugin that fails"
    plugin_stage: ClassVar[IngestStage] = "parse"

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Raise an error.

        Raises
        ------
        ValueError
            Always raised for testing.
        """
        _ = self, ctx
        msg = "Intentional failure"
        raise ValueError(msg)


@dataclass
class TableWriterPlugin(TableWriterIngestPlugin):
    """Plugin that writes to tables."""

    plugin_name: ClassVar[str] = "table_writer"
    plugin_description: ClassVar[str] = "A table writer plugin"
    plugin_stage: ClassVar[IngestStage] = "index"
    produces_tables: ClassVar[tuple[str, ...]] = ("core.functions", "core.modules")

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Return row counts for tables.

        Returns
        -------
        Mapping[str, int] | None
            Row counts by table.
        """
        _ = self, ctx
        return {"core.functions": ROW_COUNT_5, "core.modules": ROW_COUNT_3}


# =============================================================================
# Default Registry Tests (Real Plugins)
# =============================================================================


def test_registry_includes_all_expected_plugins() -> None:
    """Ensure the default registry is synchronized with the expected plugin set."""
    registry = get_ingest_registry()
    names = set(registry.list_names())
    expected = {
        "repo_scan",
        "scip_ingest",
        "cst_extract",
        "ast_extract",
        "typing_ingest",
        "coverage_ingest",
        "tests_ingest",
        "docstrings_ingest",
        "config_ingest",
    }
    missing = expected - names
    assert not missing, f"Missing ingestion plugins in registry: {sorted(missing)}"


def test_metadata_exposes_tables_and_deps() -> None:
    """Verify registry metadata surfaces dependencies and tables accurately."""
    registry = get_ingest_registry()

    repo_scan = registry.get("repo_scan")
    meta = repo_scan.metadata
    assert "core.modules" in meta.produces_tables, "repo_scan metadata missing core.modules"
    assert meta.depends_on == (), f"repo_scan depends_on should be empty, found: {meta.depends_on}"

    scip = registry.get("scip_ingest")
    scip_meta = scip.metadata
    assert "core.scip_symbols" in scip_meta.produces_tables
    assert "core.goid_crosswalk" in scip_meta.produces_tables
    assert "repo_scan" in scip_meta.depends_on, f"scip_ingest depends_on: {scip_meta.depends_on}"

    docstrings = registry.get("docstrings_ingest")
    doc_meta = docstrings.metadata
    assert "core.docstrings" in doc_meta.produces_tables
    assert "repo_scan" in doc_meta.depends_on


def test_plan_respects_dependencies() -> None:
    """Confirm plan ordering respects declared prerequisites."""
    plan = plan_ingest_plugins(
        PlanOptions(
            plugin_names=(
                "repo_scan",
                "scip_ingest",
                "ast_extract",
                "cst_extract",
                "docstrings_ingest",
            ),
            defaults=DEFAULT_INGEST_PLUGINS,
        )
    )
    order = plan.ordered_names
    positions = {name: order.index(name) for name in order}

    assert positions["repo_scan"] < positions["scip_ingest"], "repo_scan must precede scip_ingest"
    assert positions["repo_scan"] < positions["ast_extract"], "repo_scan must precede ast_extract"
    assert positions["repo_scan"] < positions["cst_extract"], "repo_scan must precede cst_extract"
    assert positions["repo_scan"] < positions["docstrings_ingest"]


def test_disabled_plugins_are_skipped() -> None:
    """Verify disabled plugins are excluded from the plan."""
    plan = plan_ingest_plugins(
        PlanOptions(
            plugin_names=DEFAULT_INGEST_PLUGINS,
            disabled=("scip_ingest", "typing_ingest"),
            defaults=DEFAULT_INGEST_PLUGINS,
        )
    )

    ordered = plan.ordered_names
    assert "scip_ingest" not in ordered, "scip_ingest should be excluded when disabled"
    assert "typing_ingest" not in ordered, "typing_ingest should be excluded when disabled"

    skipped_names = {s.name for s in plan.skipped_plugins}
    assert "scip_ingest" in skipped_names
    assert "typing_ingest" in skipped_names


def test_custom_plugin_registry_execution(tmp_path: Path) -> None:
    """Smoke test exercising dependency expansion with custom plugins."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = BuildPaths.from_repo_root(repo_root)
    snapshot = SnapshotRef.from_args(repo="demo/repo", commit="deadbeef", repo_root=repo_root)
    code_profile = default_code_profile(repo_root)
    config_profile = default_config_profile(repo_root)
    tools = ToolsConfig.default()

    executed: list[str] = []

    @dataclass
    class AlphaPlugin(BaseIngestPlugin):
        plugin_name: ClassVar[str] = "alpha"
        plugin_description: ClassVar[str] = "First step"
        plugin_stage: ClassVar[IngestStage] = "scan"
        depends_on: ClassVar[tuple[str, ...]] = ()

        def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
            _ = self, ctx
            executed.append("alpha")
            return None

    @dataclass
    class BravoPlugin(BaseIngestPlugin):
        plugin_name: ClassVar[str] = "bravo"
        plugin_description: ClassVar[str] = "Second step"
        plugin_stage: ClassVar[IngestStage] = "parse"
        depends_on: ClassVar[tuple[str, ...]] = ("alpha",)

        def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
            _ = self, ctx
            executed.append("bravo")
            return None

    @dataclass
    class CharliePlugin(BaseIngestPlugin):
        plugin_name: ClassVar[str] = "charlie"
        plugin_description: ClassVar[str] = "Final step"
        plugin_stage: ClassVar[IngestStage] = "enrich"
        depends_on: ClassVar[tuple[str, ...]] = ("bravo",)

        def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
            _ = self, ctx
            executed.append("charlie")
            return None

    registry = IngestPluginRegistry()
    registry.register(AlphaPlugin())
    registry.register(BravoPlugin())
    registry.register(CharliePlugin())

    plan = registry.plan(
        PlanOptions(
            plugin_names=("alpha", "bravo", "charlie"),
            defaults=("alpha", "bravo", "charlie"),
        )
    )

    gateway = open_ingestion_gateway()
    try:
        scratch = IngestRuntimeScratch()
        resources = ResourceRegistry()
        for plugin in plan.plugins:
            ctx = IngestExecutionContext(
                gateway=gateway,
                snapshot=snapshot,
                paths=paths,
                tools=tools,
                code_profile=code_profile,
                config_profile=config_profile,
                resources=resources,
                scratch=scratch,
                plugin_name=plugin.metadata.name,
            )
            plugin.execute(ctx)
    finally:
        gateway.close()

    assert executed == ["alpha", "bravo", "charlie"], f"Unexpected order: {executed}"


# =============================================================================
# IngestResourceHints Tests
# =============================================================================


def test_resource_hints_defaults() -> None:
    """IngestResourceHints should have sensible defaults."""
    hints = IngestResourceHints()

    assert hints.max_runtime_ms is None
    assert hints.memory_mb_hint is None
    assert hints.cpu_intensive is False
    assert hints.io_intensive is False


def test_resource_hints_custom_values() -> None:
    """IngestResourceHints should accept custom values."""
    hints = IngestResourceHints(
        max_runtime_ms=RUNTIME_MS_5000,
        memory_mb_hint=MEMORY_MB_512,
        cpu_intensive=True,
        io_intensive=False,
    )

    assert hints.max_runtime_ms == RUNTIME_MS_5000
    assert hints.memory_mb_hint == MEMORY_MB_512
    assert hints.cpu_intensive is True
    assert hints.io_intensive is False


# =============================================================================
# IngestPluginMetadata Tests
# =============================================================================


def test_plugin_metadata_required_fields() -> None:
    """IngestPluginMetadata should require name, description, stage."""
    metadata = IngestPluginMetadata(
        name="test_plugin",
        description="Test description",
        stage="parse",
    )

    assert metadata.name == "test_plugin"
    assert metadata.description == "Test description"
    assert metadata.stage == "parse"


def test_plugin_metadata_defaults() -> None:
    """IngestPluginMetadata should have sensible defaults."""
    metadata = IngestPluginMetadata(
        name="test",
        description="Test",
        stage="scan",
    )

    assert metadata.severity == "fatal"
    assert metadata.enabled_by_default is True
    assert metadata.depends_on == ()
    assert metadata.provides == ()
    assert metadata.requires == ()
    assert metadata.produces_tables == ()
    assert metadata.supports_incremental is False
    assert metadata.isolation_kind == "none"


def test_plugin_metadata_custom_dependencies() -> None:
    """IngestPluginMetadata should track dependencies."""
    metadata = IngestPluginMetadata(
        name="dependent",
        description="A dependent plugin",
        stage="enrich",
        depends_on=("ast_extract", "cst_extract"),
        requires=("ast_nodes",),
    )

    assert "ast_extract" in metadata.depends_on
    assert "cst_extract" in metadata.depends_on
    assert "ast_nodes" in metadata.requires


def test_plugin_metadata_with_resource_hints() -> None:
    """IngestPluginMetadata should include resource hints."""
    hints = IngestResourceHints(max_runtime_ms=RUNTIME_MS_3000)
    metadata = IngestPluginMetadata(
        name="heavy",
        description="Heavy plugin",
        stage="index",
        resource_hints=hints,
    )

    assert metadata.resource_hints is hints
    assert metadata.resource_hints is not None
    assert metadata.resource_hints.max_runtime_ms == RUNTIME_MS_3000


# =============================================================================
# IngestPluginResult Tests
# =============================================================================


def test_result_ok_factory() -> None:
    """IngestPluginResult.ok should create successful result."""
    result = IngestPluginResult.ok()

    assert result.success is True
    assert result.error is None
    assert result.skipped is False


def test_result_ok_with_row_counts() -> None:
    """IngestPluginResult.ok should accept row counts."""
    result = IngestPluginResult.ok(row_counts={"core.test": ROW_COUNT_42})

    assert result.success is True
    assert result.row_counts is not None
    assert result.row_counts["core.test"] == ROW_COUNT_42


def test_result_ok_with_artifacts() -> None:
    """IngestPluginResult.ok should accept artifacts."""
    artifacts = {"index": TEST_ARTIFACT_PATH}
    result = IngestPluginResult.ok(artifacts=artifacts)

    assert result.artifacts is not None
    assert result.artifacts["index"] == TEST_ARTIFACT_PATH


def test_result_fail_factory() -> None:
    """IngestPluginResult.fail should create failed result."""
    result = IngestPluginResult.fail("Something went wrong")

    assert result.success is False
    assert result.error == "Something went wrong"


def test_result_fail_with_error_kind() -> None:
    """IngestPluginResult.fail should accept error kind."""
    result = IngestPluginResult.fail("timeout", error_kind="timeout_error")

    assert result.success is False
    assert result.error == "timeout"
    assert result.error_kind == "timeout_error"


def test_result_skip_factory() -> None:
    """IngestPluginResult.skip should create skipped result."""
    result = IngestPluginResult.skip("Tool not available")

    assert result.success is True
    assert result.skipped is True
    assert result.skip_reason == "Tool not available"


# =============================================================================
# IngestRuntimeScratch Tests
# =============================================================================


def test_scratch_declare_and_consume() -> None:
    """Scratch should store and retrieve values."""
    scratch = IngestRuntimeScratch()

    scratch.declare("ast_tree", {"nodes": []})
    result = scratch.consume("ast_tree")

    assert result == {"nodes": []}


def test_scratch_consume_missing_returns_default() -> None:
    """Scratch should return default for missing keys."""
    scratch = IngestRuntimeScratch()

    result = scratch.consume("missing", "default_value")

    assert result == "default_value"


def test_scratch_has_key() -> None:
    """Scratch should check key existence."""
    scratch = IngestRuntimeScratch()

    assert scratch.has("key") is False
    scratch.declare("key", "value")
    assert scratch.has("key") is True


def test_scratch_len_and_keys() -> None:
    """Scratch should report count and keys."""
    scratch = IngestRuntimeScratch()

    scratch.declare("a", 1)
    scratch.declare("b", 2)

    expected_len = 2
    assert len(scratch) == expected_len
    keys = scratch.keys()
    assert "a" in keys
    assert "b" in keys


def test_scratch_cleanup_runs_callbacks() -> None:
    """Scratch cleanup should run registered callbacks."""
    scratch = IngestRuntimeScratch()
    called = [False]

    def cleanup_fn() -> None:
        called[0] = True

    scratch.register_cleanup(cleanup_fn)
    scratch.cleanup()

    assert called[0] is True


def test_scratch_cleanup_clears_store() -> None:
    """Scratch cleanup should clear stored values."""
    scratch = IngestRuntimeScratch()
    scratch.declare("key", "value")

    scratch.cleanup()

    assert scratch.has("key") is False
    assert len(scratch) == 0


def test_scratch_cleanup_handles_callback_errors() -> None:
    """Scratch cleanup should not fail on callback errors."""
    scratch = IngestRuntimeScratch()

    def failing_callback() -> None:
        msg = "Cleanup failed"
        raise ValueError(msg)

    scratch.register_cleanup(failing_callback)

    # Should not raise
    scratch.cleanup()


# =============================================================================
# IngestPluginSkip Tests
# =============================================================================


def test_plugin_skip_basic() -> None:
    """IngestPluginSkip should capture skip info."""
    skip = IngestPluginSkip(name="slow_plugin", reason="disabled")

    assert skip.name == "slow_plugin"
    assert skip.reason == "disabled"


# =============================================================================
# IngestPluginPlan Tests
# =============================================================================


def test_plugin_plan_ordered_names() -> None:
    """IngestPluginPlan should return ordered plugin names."""
    plugin = SamplePlugin()
    plan = IngestPluginPlan(
        plugins=(plugin,),
        plan_id="test-plan",
    )

    assert plan.ordered_names == ("sample_plugin",)


def test_plugin_plan_with_skipped() -> None:
    """IngestPluginPlan should track skipped plugins."""
    skip = IngestPluginSkip(name="slow", reason="disabled")
    plan = IngestPluginPlan(
        plugins=(),
        plan_id="test",
        skipped_plugins=(skip,),
    )

    assert len(plan.skipped_plugins) == 1
    assert plan.skipped_plugins[0].name == "slow"


def test_plugin_plan_plugin_count() -> None:
    """IngestPluginPlan should track plugin count."""
    plugin = SamplePlugin()
    plan = IngestPluginPlan(
        plugins=(plugin,),
        plan_id="test",
    )

    assert len(plan.plugins) == 1


# =============================================================================
# is_ingest_plugin Type Guard Tests
# =============================================================================


def test_is_ingest_plugin_true_for_valid() -> None:
    """is_ingest_plugin should return True for valid plugins."""
    plugin = SamplePlugin()

    assert is_ingest_plugin(plugin) is True


def test_is_ingest_plugin_false_for_non_plugin() -> None:
    """is_ingest_plugin should return False for non-plugins."""

    class NotAPlugin:
        pass

    obj = NotAPlugin()

    assert is_ingest_plugin(obj) is False


def test_is_ingest_plugin_false_for_none() -> None:
    """is_ingest_plugin should return False for None."""
    assert is_ingest_plugin(None) is False


# =============================================================================
# BaseIngestPlugin Tests
# =============================================================================


def test_base_plugin_metadata_property() -> None:
    """BaseIngestPlugin should synthesize metadata from class vars."""
    plugin = SamplePlugin()
    metadata = plugin.metadata

    assert metadata.name == "sample_plugin"
    assert metadata.description == "A sample test plugin"
    assert metadata.stage == "parse"


def test_base_plugin_implements_protocol() -> None:
    """BaseIngestPlugin should implement IngestPluginProtocol."""
    plugin = SamplePlugin()

    assert isinstance(plugin, IngestPluginProtocol)


# =============================================================================
# TableWriterIngestPlugin Tests
# =============================================================================


def test_table_writer_plugin_produces_tables() -> None:
    """TableWriterIngestPlugin should track produced tables."""
    plugin = TableWriterPlugin()

    assert "core.functions" in plugin.produces_tables
    assert "core.modules" in plugin.produces_tables


# =============================================================================
# IngestPluginRegistry Tests
# =============================================================================


def test_registry_register_and_get() -> None:
    """Registry should register and retrieve plugins."""
    registry = IngestPluginRegistry()
    plugin = SamplePlugin()

    registry.register(plugin)

    retrieved = registry.get("sample_plugin")
    assert retrieved is plugin


def test_registry_get_nonexistent_raises_key_error() -> None:
    """Registry should raise KeyError for unknown plugins."""
    registry = IngestPluginRegistry()

    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_registry_plan_basic() -> None:
    """Registry should create plans from plugin names."""
    registry = IngestPluginRegistry()
    plugin = SamplePlugin()
    registry.register(plugin)

    plan = registry.plan(PlanOptions(plugin_names=["sample_plugin"]))

    assert len(plan.plugins) == 1
    assert plan.plugins[0] is plugin


def test_registry_plan_disabled() -> None:
    """Registry should skip disabled plugins in plan."""
    registry = IngestPluginRegistry()
    plugin = SamplePlugin()
    registry.register(plugin)

    plan = registry.plan(
        PlanOptions(
            plugin_names=["sample_plugin"],
            disabled=("sample_plugin",),
        )
    )

    assert len(plan.plugins) == 0
    assert any(s.name == "sample_plugin" for s in plan.skipped_plugins)


def test_registry_list_plugins() -> None:
    """Registry should list all registered plugins."""
    registry = IngestPluginRegistry()
    plugin1 = SamplePlugin()
    plugin2 = TableWriterPlugin()
    registry.register(plugin1)
    registry.register(plugin2)

    plugins = list(registry.list_all())

    plugin_names = [p.metadata.name for p in plugins]
    assert "sample_plugin" in plugin_names
    assert "table_writer" in plugin_names


# =============================================================================
# Type Literal Parametrized Tests
# =============================================================================


@pytest.mark.parametrize(
    "stage",
    ["scan", "parse", "index", "enrich", "validate"],
)
def test_ingest_stage_values(stage: IngestStage) -> None:
    """IngestStage should accept valid literal values."""
    metadata = IngestPluginMetadata(
        name="test",
        description="Test",
        stage=stage,
    )
    assert metadata.stage == stage


@pytest.mark.parametrize(
    "severity",
    ["fatal", "soft_fail", "skip_on_error"],
)
def test_ingest_severity_values(severity: IngestSeverity) -> None:
    """IngestSeverity should accept valid literal values."""
    metadata = IngestPluginMetadata(
        name="test",
        description="Test",
        stage="parse",
        severity=severity,
    )
    assert metadata.severity == severity


@pytest.mark.parametrize(
    "isolation",
    ["process", "thread", "none"],
)
def test_ingest_isolation_values(isolation: IngestIsolationKind) -> None:
    """IngestIsolationKind should accept valid literal values."""
    metadata = IngestPluginMetadata(
        name="test",
        description="Test",
        stage="parse",
        isolation_kind=isolation,
    )
    assert metadata.isolation_kind == isolation
