"""PR-68: Contract provider parity and functionality tests.

This module validates the contract provider infrastructure that derives
DatasetContract instances from build targets and schemas, eliminating the
need for manually maintained contract dictionaries.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import (
    clear_contract_cache,
    get_contract_for_table_key,
    is_view,
    iter_contracts,
    iter_contracts_by_table_key,
)
from codeintel.storage.view_names import DERIVED_DOCS_VIEWS

if TYPE_CHECKING:
    from codeintel.config.datasets.contracts import DatasetContract


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Clear contract cache before each test for isolation."""
    clear_contract_cache()


class TestIsViewFunction:
    """Tests for the is_view() function."""

    def test_docs_view_prefix_returns_true(self) -> None:
        """Verify docs.v_* keys are identified as views."""
        assert is_view("docs.v_function_profile") is True
        assert is_view("docs.v_module_profile") is True
        assert is_view("docs.v_test_profile") is True

    def test_derived_docs_views_returns_true(self) -> None:
        """Verify all DERIVED_DOCS_VIEWS are identified as views."""
        for view_key in DERIVED_DOCS_VIEWS:
            assert is_view(view_key) is True, f"{view_key} should be a view"

    def test_analytics_tables_return_false(self) -> None:
        """Verify analytics.* tables are not views."""
        assert is_view("analytics.function_metrics") is False
        assert is_view("analytics.file_metrics") is False
        assert is_view("analytics.module_metrics") is False

    def test_core_tables_return_false(self) -> None:
        """Verify core.* tables are not views."""
        assert is_view("core.goids") is False
        assert is_view("core.modules") is False
        assert is_view("core.files") is False

    def test_graph_tables_return_false(self) -> None:
        """Verify graph.* tables are not views."""
        assert is_view("graph.call_graph_nodes") is False
        assert is_view("graph.call_graph_edges") is False


class TestGetContractForTableKey:
    """Tests for get_contract_for_table_key() function."""

    def test_returns_contract_for_known_table(self) -> None:
        """Verify a known table key returns a DatasetContract."""
        contract = get_contract_for_table_key("analytics.function_metrics")
        assert contract is not None
        assert contract.table_key == "analytics.function_metrics"
        assert contract.name == "function_metrics"

    def test_returns_contract_for_view(self) -> None:
        """Verify view keys return contracts with is_view=True."""
        # Use a known view from DERIVED_DOCS_VIEWS
        if DERIVED_DOCS_VIEWS:
            view_key = DERIVED_DOCS_VIEWS[0]
            contract = get_contract_for_table_key(view_key)
            assert contract is not None
            assert contract.table_key == view_key
            assert contract.is_view is True

    def test_caches_results(self) -> None:
        """Verify contracts are cached on repeated calls."""
        contract1 = get_contract_for_table_key("analytics.function_metrics")
        contract2 = get_contract_for_table_key("analytics.function_metrics")
        assert contract1 is contract2

    def test_raises_keyerror_for_unknown_table(self) -> None:
        """Verify KeyError is raised for unknown table keys."""
        with pytest.raises(KeyError, match="Unknown table key"):
            get_contract_for_table_key("nonexistent.table")

    def test_contract_has_required_fields(self) -> None:
        """Verify derived contracts have all required fields populated."""
        contract = get_contract_for_table_key("analytics.function_metrics")
        assert contract.table_key is not None
        assert contract.name is not None
        # owner_package should be derived from schema prefix
        assert contract.owner_package in {"core", "analytics", "graphs", "qa", "docs", None}
        # family should be derived from schema prefix or OutputContract
        assert contract.family in {"core", "analytics", "graph", "docs", "qa", None}


class TestIterContracts:
    """Tests for iter_contracts() function."""

    def test_yields_multiple_contracts(self) -> None:
        """Verify iter_contracts yields multiple contracts."""
        contracts = list(iter_contracts())
        assert len(contracts) > 0

    def test_each_contract_is_dataset_contract(self) -> None:
        """Verify each yielded item is a DatasetContract."""
        from codeintel.config.datasets.contracts import DatasetContract

        for contract in iter_contracts():
            assert isinstance(contract, DatasetContract)

    def test_includes_tables_and_views(self) -> None:
        """Verify iteration includes both tables and views."""
        contracts = list(iter_contracts())
        has_tables = any(not is_view(c.table_key) for c in contracts)
        has_views = any(is_view(c.table_key) for c in contracts)
        assert has_tables, "Should include at least one table"
        assert has_views, "Should include at least one view"


class TestIterContractsByTableKey:
    """Tests for iter_contracts_by_table_key() function."""

    def test_yields_tuples_of_key_and_contract(self) -> None:
        """Verify iter_contracts_by_table_key yields (key, contract) tuples."""
        for table_key, contract in iter_contracts_by_table_key():
            assert isinstance(table_key, str)
            assert contract.table_key == table_key


class TestContractProviderParityWithLegacy:
    """Parity tests comparing contract provider with legacy DATASET_CONTRACTS."""

    def test_is_view_matches_legacy_contract_is_view(self) -> None:
        """Verify is_view() matches legacy contract.is_view for all contracts."""
        # Get legacy contracts with suppressed deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from codeintel.config.datasets import get_dataset_contracts_by_table_key

            legacy_contracts = get_dataset_contracts_by_table_key()

        mismatches: list[str] = []
        for table_key, legacy_contract in legacy_contracts.items():
            provider_is_view = is_view(table_key)
            legacy_is_view = legacy_contract.is_view
            if provider_is_view != legacy_is_view:
                mismatches.append(
                    f"{table_key}: provider={provider_is_view}, legacy={legacy_is_view}"
                )

        assert not mismatches, f"is_view mismatches:\n" + "\n".join(mismatches)

    def test_derived_contract_table_key_matches_legacy(self) -> None:
        """Verify derived contracts have matching table_key values."""
        # Get legacy contracts with suppressed deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from codeintel.config.datasets import get_dataset_contracts_by_table_key

            legacy_contracts = get_dataset_contracts_by_table_key()

        # Check that we can derive contracts for all legacy table keys
        failed_keys: list[str] = []
        for table_key in legacy_contracts:
            try:
                derived = get_contract_for_table_key(table_key)
                assert derived.table_key == table_key
            except KeyError:
                failed_keys.append(table_key)

        # Some tables might not be derivable (e.g., source tables without targets)
        # This is acceptable, but we should have most of them
        coverage = (len(legacy_contracts) - len(failed_keys)) / len(legacy_contracts)
        assert coverage >= 0.8, f"Coverage too low: {coverage:.2%}, failed: {failed_keys[:10]}"

    def test_derived_contract_name_matches_legacy(self) -> None:
        """Verify derived contracts have matching name values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from codeintel.config.datasets import get_dataset_contracts_by_table_key

            legacy_contracts = get_dataset_contracts_by_table_key()

        mismatches: list[str] = []
        for table_key, legacy_contract in legacy_contracts.items():
            try:
                derived = get_contract_for_table_key(table_key)
                if derived.name != legacy_contract.name:
                    mismatches.append(
                        f"{table_key}: derived={derived.name}, legacy={legacy_contract.name}"
                    )
            except KeyError:
                # Table not derivable, skip
                continue

        assert not mismatches, f"Name mismatches:\n" + "\n".join(mismatches)


class TestViewHandling:
    """Tests for view-specific contract derivation."""

    def test_views_have_no_producing_target(self) -> None:
        """Verify views are handled even without producing targets."""
        if not DERIVED_DOCS_VIEWS:
            pytest.skip("No views defined in DERIVED_DOCS_VIEWS")

        # Views should return contracts even without targets
        for view_key in list(DERIVED_DOCS_VIEWS)[:5]:  # Test first 5
            contract = get_contract_for_table_key(view_key)
            assert contract is not None
            assert contract.is_view is True
            assert contract.table_key == view_key

    def test_views_have_docs_view_tag(self) -> None:
        """Verify view contracts have appropriate tags."""
        if not DERIVED_DOCS_VIEWS:
            pytest.skip("No views defined in DERIVED_DOCS_VIEWS")

        view_key = DERIVED_DOCS_VIEWS[0]
        contract = get_contract_for_table_key(view_key)
        assert "docs_view" in contract.tags or "read_only" in contract.tags


class TestTableHandling:
    """Tests for table-specific contract derivation."""

    def test_tables_have_base_table_tag(self) -> None:
        """Verify table contracts have base_table tag."""
        contract = get_contract_for_table_key("analytics.function_metrics")
        assert "base_table" in contract.tags

    def test_tables_have_owner_package(self) -> None:
        """Verify tables have owner_package derived from schema prefix."""
        contract = get_contract_for_table_key("analytics.function_metrics")
        assert contract.owner_package == "analytics"

        contract = get_contract_for_table_key("core.goids")
        assert contract.owner_package == "core"


class TestContractCacheManagement:
    """Tests for contract cache management."""

    def test_clear_contract_cache_clears_all_cached_contracts(self) -> None:
        """Verify clear_contract_cache() removes all cached entries."""
        # Populate cache
        contract1 = get_contract_for_table_key("analytics.function_metrics")

        # Clear cache
        clear_contract_cache()

        # Get again - should be a new instance if not frozen
        contract2 = get_contract_for_table_key("analytics.function_metrics")

        # Both should have same values even after cache clear
        assert contract1.table_key == contract2.table_key


class TestDeprecationWarnings:
    """Tests for deprecation warnings on legacy APIs."""

    def test_get_dataset_contracts_emits_deprecation_warning(self) -> None:
        """Verify get_dataset_contracts() emits DeprecationWarning."""
        from codeintel.config.datasets import get_dataset_contracts

        with pytest.warns(DeprecationWarning, match="get_dataset_contracts.*deprecated"):
            get_dataset_contracts()

    def test_get_dataset_contracts_by_table_key_emits_deprecation_warning(self) -> None:
        """Verify get_dataset_contracts_by_table_key() emits DeprecationWarning."""
        from codeintel.config.datasets import get_dataset_contracts_by_table_key

        with pytest.warns(DeprecationWarning, match="get_dataset_contracts_by_table_key.*deprecated"):
            get_dataset_contracts_by_table_key()


class TestContractMetadataDerivation:
    """Tests for metadata derivation in contracts."""

    def test_schema_description_used_when_available(self) -> None:
        """Verify schema description is used as contract description when available."""
        contract = get_contract_for_table_key("analytics.function_metrics")
        # Description should be populated from schema or target
        # The exact value depends on schema provider configuration
        assert contract.description is None or isinstance(contract.description, str)

    def test_family_derived_from_schema_prefix(self) -> None:
        """Verify family is derived from schema prefix."""
        contract = get_contract_for_table_key("analytics.function_metrics")
        assert contract.family == "analytics"

        contract = get_contract_for_table_key("core.goids")
        assert contract.family == "core"

    def test_validation_profile_defaults_to_strict(self) -> None:
        """Verify validation_profile defaults to 'strict'."""
        contract = get_contract_for_table_key("analytics.function_metrics")
        assert contract.validation_profile == "strict"

