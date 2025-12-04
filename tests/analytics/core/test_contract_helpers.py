"""Tests for contract helper utilities."""

from __future__ import annotations

from codeintel.analytics.core.contracts import (
    OutputContractSpec,
    build_plugin_output_contracts,
)
from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import (
    PluginMetadata,
    PluginOutputSpec,
    PluginResult,
    ValidationResult,
)


class _StubPlugin:
    def __init__(
        self,
        *,
        metadata: PluginMetadata,
        explicit_contracts: tuple[OutputContractSpec, ...] | None = None,
    ) -> None:
        self._metadata = metadata
        if explicit_contracts is not None:
            self.output_contracts = explicit_contracts

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


def test_builds_contracts_from_metadata_outputs() -> None:
    """Derive contracts from plugin metadata outputs."""
    plugin = _StubPlugin(
        metadata=PluginMetadata(
            name="stub",
            description="stub plugin",
            kind="analytics",
            stage="graph",
            outputs=(
                PluginOutputSpec(
                    name="metrics",
                    tables=("analytics.metrics",),
                    min_rows=1,
                    required_columns=("repo", "commit"),
                ),
                PluginOutputSpec(
                    name="artifact_only",
                    tables=(),
                    artifact_type="json",
                ),
            ),
        )
    )

    contracts = build_plugin_output_contracts(plugin)

    assert len(contracts) == 1
    plugin_contract = contracts[0]
    assert plugin_contract.plugin_name == "stub"
    assert len(plugin_contract.contracts) == 1
    contract = plugin_contract.contracts[0]
    assert contract.table == "analytics.metrics"
    assert contract.min_rows == 1
    assert contract.required_columns == ("repo", "commit")


def test_prefers_explicit_output_contracts() -> None:
    """Prefer explicit output_contracts while retaining metadata-derived specs."""
    explicit = OutputContractSpec(
        table="analytics.explicit",
        min_rows=2,
        required_columns=("repo",),
    )
    plugin = _StubPlugin(
        metadata=PluginMetadata(
            name="stub_explicit",
            description="stub plugin",
            kind="analytics",
            stage="graph",
            outputs=(
                PluginOutputSpec(
                    name="metadata_metrics",
                    tables=("analytics.metadata",),
                    min_rows=1,
                    required_columns=("repo", "commit"),
                ),
            ),
        ),
        explicit_contracts=(explicit,),
    )

    contracts = build_plugin_output_contracts(plugin)

    assert len(contracts) == 1
    plugin_contract = contracts[0]
    assert plugin_contract.plugin_name == "stub_explicit"
    contract_tables = {contract.table for contract in plugin_contract.contracts}
    assert "analytics.explicit" in contract_tables
    assert "analytics.metadata" in contract_tables


def test_returns_empty_when_no_contracts_available() -> None:
    """Return empty tuple when neither explicit nor metadata contracts are present."""
    plugin = _StubPlugin(
        metadata=PluginMetadata(
            name="empty",
            description="no contracts",
            kind="analytics",
            stage="graph",
        )
    )

    assert build_plugin_output_contracts(plugin) == ()
