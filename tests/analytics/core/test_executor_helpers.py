"""Tests for RecipeExecutor helper methods."""

from __future__ import annotations

import tempfile
from pathlib import Path

from codeintel.analytics.core.context import PluginExecutionContext, PluginScratch
from codeintel.analytics.core.protocol import (
    AnalyticsPluginProtocol,
    PluginMetadata,
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.recipes.executor import RecipeExecutionContext, RecipeExecutor
from codeintel.analytics.recipes.model import Recipe, RecipeScope
from codeintel.config.primitives import SnapshotRef
from tests._helpers.gateway import open_ingestion_gateway


class _ValidationFailurePlugin(AnalyticsPluginProtocol):
    """Plugin that always fails validation."""

    def __init__(self) -> None:
        self._meta = PluginMetadata(
            name="invalid.plugin",
            description="Invalid plugin",
            kind="analytics",
            stage="function",
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._meta

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        del ctx
        return ValidationResult.failure((f"{self._meta.name} bad inputs",))

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        del ctx
        return PluginResult.ok(meta={"plugin": self._meta.name})


class _ErroringPlugin(AnalyticsPluginProtocol):
    """Plugin that raises during execution."""

    def __init__(self) -> None:
        self._meta = PluginMetadata(
            name="error.plugin",
            description="Erroring plugin",
            kind="analytics",
            stage="function",
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._meta

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        del ctx
        _ = self._meta
        return ValidationResult.success()

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        del ctx
        message = f"{self._meta.name} boom"
        raise RuntimeError(message)


class _SuccessPlugin(AnalyticsPluginProtocol):
    """Plugin that succeeds and returns row counts."""

    def __init__(self) -> None:
        self._meta = PluginMetadata(
            name="success.plugin",
            description="Success plugin",
            kind="analytics",
            stage="function",
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._meta

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        del ctx
        _ = self._meta
        return ValidationResult.success()

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        del ctx
        return PluginResult.ok(
            row_counts={"table": 5},
            meta={"source": self._meta.name},
        )


def _context(scope: RecipeScope | None = None) -> RecipeExecutionContext:
    gateway = open_ingestion_gateway()
    return RecipeExecutionContext(
        gateway=gateway,
        snapshot=SnapshotRef(
            repo="test-repo",
            commit="abc123",
            repo_root=Path(tempfile.gettempdir()),
        ),
        scope=scope or RecipeScope(),
    )


def test_merge_configs_merges_without_mutation() -> None:
    """merge_configs should combine defaults and overrides immutably."""
    recipe = Recipe(
        name="base",
        description="",
        plugins=("p1",),
        default_configs={"p1": {"base": True}},
    )
    overrides = {"p1": {"override": 1}}

    merged = RecipeExecutor.merge_configs(recipe, overrides)

    assert merged["p1"] == {"base": True, "override": 1}
    assert recipe.default_configs == {"p1": {"base": True}}
    assert overrides == {"p1": {"override": 1}}


def test_execute_plugin_handles_validation_failure() -> None:
    """Validation failure should produce a failed record with error message."""
    plugin = _ValidationFailurePlugin()
    context = _context()
    scratch = PluginScratch()

    record = RecipeExecutor.execute_plugin(
        plugin=plugin,
        context=context,
        config={},
        run_id="run-1",
        scratch=scratch,
    )

    assert record.status == "failed"
    assert "Validation failed" in (record.error or "")
    assert record.plugin_name == plugin.metadata.name


def test_execute_plugin_handles_execution_error() -> None:
    """Runtime errors should be captured as failed plugin records."""
    plugin = _ErroringPlugin()
    context = _context()
    scratch = PluginScratch()

    record = RecipeExecutor.execute_plugin(
        plugin=plugin,
        context=context,
        config={},
        run_id="run-2",
        scratch=scratch,
    )

    assert record.status == "failed"
    assert "boom" in (record.error or "")
    assert record.row_counts == {}


def test_execute_plugin_records_success() -> None:
    """Successful execution should capture row counts and metadata."""
    plugin = _SuccessPlugin()
    context = _context()
    scratch = PluginScratch()

    record = RecipeExecutor.execute_plugin(
        plugin=plugin,
        context=context,
        config={},
        run_id="run-3",
        scratch=scratch,
    )

    assert record.status == "succeeded"
    assert record.row_counts == {"table": 5}
    assert record.meta.get("source") == "success.plugin"
