"""Tests for contract caching in the plugin executor."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.core.context import (
    PluginExecutionContext,
    PluginExecutionContextBuilder,
)
from codeintel.analytics.core.contracts import OutputContractSpec
from codeintel.analytics.core.executor import PluginExecutor
from codeintel.analytics.core.protocol import (
    PluginMetadata,
    PluginOutputSpec,
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.core.registry import PluginPlan
from codeintel.config.primitives import SnapshotRef
from tests._helpers.gateway import open_fresh_duckdb, seed_tables


class _ValidatedStubPlugin:
    def __init__(
        self,
        *,
        name: str = "stub",
        contracts: tuple[OutputContractSpec, ...] | None = None,
        outputs: tuple[PluginOutputSpec, ...] = (),
    ) -> None:
        self.output_contracts = contracts or ()
        self._metadata = PluginMetadata(
            name=name,
            description="stub",
            kind="analytics",
            stage="graph",
            outputs=outputs,
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @staticmethod
    def execute(ctx: PluginExecutionContext) -> PluginResult:
        _ = ctx
        return PluginResult.ok()

    @staticmethod
    def validate_inputs(ctx: PluginExecutionContext) -> ValidationResult:
        _ = ctx
        return ValidationResult.success()


def test_contracts_cached_per_plugin_name() -> None:
    """Contracts should be built once and reused per plugin name."""
    output_spec = PluginOutputSpec(
        name="metrics",
        tables=("analytics.metrics",),
        min_rows=1,
        required_columns=("repo", "commit"),
    )
    plugin = _ValidatedStubPlugin(outputs=(output_spec,))
    executor = PluginExecutor()

    first = executor.get_plugin_contracts(plugin)
    second = executor.get_plugin_contracts(plugin)

    assert first is second
    assert len(first) == 1


def test_contracts_empty_when_no_outputs_or_explicit() -> None:
    """Return empty contracts when plugin declares none."""
    plugin = _ValidatedStubPlugin(outputs=())
    executor = PluginExecutor()

    contracts = executor.get_plugin_contracts(plugin)

    assert contracts == ()


class _CountingPlugin:
    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            name="counting",
            description="counts accesses",
            kind="analytics",
            stage="graph",
        )
        self.access_count = 0

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @property
    def output_contracts(self) -> tuple[OutputContractSpec, ...]:
        self.access_count += 1
        if self.access_count > 1:
            msg = "output_contracts accessed more than once"
            raise RuntimeError(msg)
        return (
            OutputContractSpec(
                table="analytics.contracts",
                min_rows=0,
            ),
        )

    @staticmethod
    def execute(ctx: PluginExecutionContext) -> PluginResult:
        _ = ctx
        return PluginResult.ok()

    @staticmethod
    def validate_inputs(ctx: PluginExecutionContext) -> ValidationResult:
        _ = ctx
        return ValidationResult.success()


def test_contracts_not_rebuilt_after_cache_fill() -> None:
    """Second cache lookup should not re-evaluate output_contracts."""
    plugin = _CountingPlugin()
    executor = PluginExecutor()

    first = executor.get_plugin_contracts(plugin)
    second = executor.get_plugin_contracts(plugin)

    assert first is second
    assert plugin.access_count == 1


def test_execute_populates_contract_results_for_metadata_contracts(tmp_path: Path) -> None:
    """Executor validates metadata-derived contracts using real DuckDB gateway."""
    output_spec = PluginOutputSpec(
        name="contracts",
        tables=("analytics.contracts",),
        min_rows=1,
        required_columns=("repo", "commit"),
    )
    plugin = _ValidatedStubPlugin(outputs=(output_spec,), name="with_contracts")
    gateway = open_fresh_duckdb(tmp_path / "contracts.duckdb")
    try:
        seed_tables(
            gateway,
            [
                "CREATE SCHEMA IF NOT EXISTS analytics",
                "DROP TABLE IF EXISTS analytics.contracts",
                """
                CREATE TABLE analytics.contracts (
                    repo TEXT,
                    commit TEXT,
                    value INTEGER
                )
                """,
            ],
        )
        gateway.con.execute(
            "INSERT INTO analytics.contracts (repo, commit, value) VALUES (?, ?, ?)",
            ["repo", "deadbeef", 1],
        )
        snapshot = SnapshotRef(repo="repo", commit="deadbeef", repo_root=tmp_path)
        builder = PluginExecutionContextBuilder(
            gateway=gateway,
            snapshot=snapshot,
            run_id="run-1",
        )
        ctx = builder.build()
        plan = PluginPlan(plugins=(plugin,))
        executor = PluginExecutor()

        report = executor.execute(ctx, plan)
    finally:
        gateway.close()

    assert report.contract_results["with_contracts"].valid is True
