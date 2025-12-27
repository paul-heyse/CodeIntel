"""Contract resolution seams for metadata injection and caching."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.contracts import OutputContract
from codeintel.build.schemas import (
    ContractResolutionSettings,
    clear_contract_cache,
    get_contract_for_table_key,
    iter_contracts,
)
from codeintel.build.target_metadata import (
    TargetMetadataProvider,
    clear_target_metadata_cache,
    is_target_metadata_loaded,
)
from codeintel.build.targets import TargetDescriptor
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_false, expect_true
from tests._helpers.catalog import make_target_descriptor
from tests._helpers.contracts import table_output_for_key


@dataclass(frozen=True)
class _StubTargetMetadataProvider(TargetMetadataProvider):
    target: TargetDescriptor

    def get_target(self, name: str) -> TargetDescriptor | None:
        if name == self.target.name:
            return self.target
        return None

    def target_for_table_key(self, table_key: str) -> TargetDescriptor | None:
        if table_key in self.target.table_keys:
            return self.target
        return None

    def target_for_artifact(self, artifact_name: str) -> TargetDescriptor | None:
        if artifact_name in self.target.contract.artifact_names:
            return self.target
        return None


def test_iter_contracts_initializes_target_metadata() -> None:
    """Verify full contract enumeration initializes target metadata."""
    clear_target_metadata_cache()
    clear_contract_cache()
    expect_false(is_target_metadata_loaded())

    _ = list(iter_contracts())

    expect_true(is_target_metadata_loaded())


def test_contract_resolution_uses_injected_target_metadata_provider() -> None:
    """Verify injected TargetMetadataProvider overrides are applied."""
    clear_target_metadata_cache()
    clear_contract_cache()

    table_key = "analytics.function_metrics"
    contract = OutputContract(
        tables=(table_output_for_key(table_key),),
        owner="unit-test-owner",
        jsonl_filenames=("custom.jsonl",),
    )
    target = make_target_descriptor(
        name="unit_test_target",
        module="analytics",
        contract=contract,
    )
    provider = _StubTargetMetadataProvider(target=target)

    settings = ContractResolutionSettings(target_metadata_provider=provider)
    resolved = get_contract_for_table_key(table_key, settings=settings)
    expect_equal(resolved.owner, "unit-test-owner")
    expect_equal(resolved.jsonl_filename, "custom.jsonl")
    expect_false(is_target_metadata_loaded())
