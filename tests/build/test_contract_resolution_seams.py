"""Contract resolution seams for declared-only enumeration and overrides."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.contracts import OutputContract
from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    clear_contract_cache,
    get_contract_for_table_key,
    iter_contracts,
)
from codeintel.build.target_metadata import (
    OutputInventory,
    TargetMetadataProvider,
    clear_target_metadata_cache,
    is_target_metadata_loaded,
)
from codeintel.build.targets import OutputTarget
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_false, expect_true
from tests._helpers.contracts import table_schema_for_key


@dataclass(frozen=True)
class _StubTargetMetadataProvider(TargetMetadataProvider):
    target: OutputTarget

    def get_target(self, name: str) -> OutputTarget | None:
        if name == self.target.name:
            return self.target
        return None

    def target_for_table_key(self, table_key: str) -> OutputTarget | None:
        if table_key in self.target.table_keys:
            return self.target
        return None

    def target_for_artifact(self, artifact_name: str) -> OutputTarget | None:
        if artifact_name in self.target.contract.artifact_names:
            return self.target
        return None


def test_schema_only_contracts_do_not_load_target_metadata() -> None:
    """Verify declared-only contract enumeration avoids target metadata."""
    clear_target_metadata_cache()
    clear_contract_cache()
    expect_false(is_target_metadata_loaded())

    _ = list(
        iter_contracts(
            settings=ContractResolutionSettings(mode=ContractResolutionMode.DECLARED_ONLY)
        )
    )

    expect_false(is_target_metadata_loaded())


def test_contract_resolution_uses_injected_target_metadata_provider() -> None:
    """Verify injected TargetMetadataProvider overrides are applied."""
    clear_target_metadata_cache()
    clear_contract_cache()

    table_key = "analytics.function_metrics"
    contract = OutputContract(
        tables=(table_schema_for_key(table_key),),
        owner="unit-test-owner",
        jsonl_filenames=("custom.jsonl",),
    )
    target = OutputTarget(name="unit_test_target", module="analytics", contract=contract)
    provider = _StubTargetMetadataProvider(target=target)

    settings = ContractResolutionSettings(
        mode=ContractResolutionMode.FULL,
        target_metadata_provider=provider,
    )
    resolved = get_contract_for_table_key(table_key, settings=settings)
    expect_equal(resolved.owner, "unit-test-owner")
    expect_equal(resolved.jsonl_filename, "custom.jsonl")
    expect_false(is_target_metadata_loaded())


def test_contract_resolution_honors_output_inventory() -> None:
    """Verify output inventory exclusions are respected."""
    clear_contract_cache()
    inventory = OutputInventory(
        datasets_by_target={"stub": ("analytics.function_metrics",)},
        artifacts_by_target={},
    )
    settings = ContractResolutionSettings(
        mode=ContractResolutionMode.DECLARED_ONLY,
        output_inventory=inventory,
    )

    table_keys = {contract.table_key for contract in iter_contracts(settings=settings)}
    expect_true(table_keys)
    expect_false("analytics.function_metrics" in table_keys)
