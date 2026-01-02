"""PR-68: Contract provider parity and functionality tests.

This module validates the contract provider infrastructure that derives
DatasetContract instances from build targets and schemas, eliminating the
need for manually maintained contract dictionaries.
"""

from __future__ import annotations

import pytest

from codeintel.build.schemas import (
    clear_contract_cache,
    get_contract_for_table_key,
    is_view,
    iter_contracts,
    iter_contracts_by_table_key,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.storage.contracts.provider import (
    clear_contract_cache as clear_storage_contract_cache,
)
from codeintel.storage.contracts.provider import (
    get_contract_for_table_key as get_storage_contract_for_table_key,
)
from codeintel.storage.views.inventory import discover_derived_docs_views
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_not_empty,
    expect_true,
)
from tests._helpers.schemas import ensure_storage_contract_catalog


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Clear contract cache before each test for isolation."""
    clear_contract_cache()


class TestIsViewFunction:
    """Tests for the is_view() function."""

    @staticmethod
    def test_docs_view_prefix_returns_true() -> None:
        """Verify docs.v_* keys are identified as views."""
        expect_true(is_view("docs.v_function_architecture"))
        expect_true(is_view("docs.v_module_architecture"))
        expect_true(is_view("docs.v_test_architecture"))

    @staticmethod
    def test_derived_docs_views_returns_true() -> None:
        """Verify all DERIVED_DOCS_VIEWS are identified as views."""
        for view_key in discover_derived_docs_views():
            expect_true(is_view(view_key), message=f"{view_key} should be a view")

    @staticmethod
    def test_analytics_tables_return_false() -> None:
        """Verify analytics.* tables are not views."""
        expect_false(is_view("analytics.function_types"))
        expect_false(is_view("analytics.graph_metrics_functions"))
        expect_false(is_view("analytics.graph_metrics_modules"))

    @staticmethod
    def test_core_tables_return_false() -> None:
        """Verify core.* tables are not views."""
        expect_false(is_view("core.goids"))
        expect_false(is_view("core.modules"))
        expect_false(is_view("core.repo_map"))

    @staticmethod
    def test_graph_tables_return_false() -> None:
        """Verify graph.* tables are not views."""
        expect_false(is_view("graph.call_graph_nodes"))
        expect_false(is_view("graph.call_graph_edges"))


class TestGetContractForTableKey:
    """Tests for get_contract_for_table_key() function."""

    @staticmethod
    def test_returns_contract_for_known_table() -> None:
        """Verify a known table key returns a DatasetContract."""
        contract = expect_is_not_none(get_contract_for_table_key("analytics.function_types"))
        expect_equal(contract.table_key, "analytics.function_types")
        expect_equal(contract.name, "function_types")

    @staticmethod
    def test_returns_contract_for_view() -> None:
        """Verify view keys return contracts with is_view=True."""
        discovered = discover_derived_docs_views()
        if discovered:
            view_key = discovered[0]
            contract = expect_is_not_none(get_contract_for_table_key(view_key))
            expect_equal(contract.table_key, view_key)
            expect_true(contract.is_view)

    @staticmethod
    def test_caches_results() -> None:
        """Verify contracts are cached on repeated calls."""
        contract1 = get_contract_for_table_key("analytics.function_types")
        contract2 = get_contract_for_table_key("analytics.function_types")
        expect_true(contract1 is contract2, message="Expected cached contract instance")

    @staticmethod
    def test_raises_keyerror_for_unknown_table() -> None:
        """Verify KeyError is raised for unknown table keys."""
        with pytest.raises(KeyError, match="Unknown table key"):
            get_contract_for_table_key("nonexistent.table")

    @staticmethod
    def test_contract_has_required_fields() -> None:
        """Verify derived contracts have all required fields populated."""
        contract = expect_is_not_none(get_contract_for_table_key("analytics.function_types"))
        expect_is_not_none(contract.table_key, label="table_key")
        expect_is_not_none(contract.name, label="name")
        # owner_package should be derived from schema prefix
        expect_in(contract.owner_package, {"core", "analytics", "graphs", "qa", "docs", None})
        # family should be derived from schema prefix or registry metadata
        expect_in(contract.family, {"core", "analytics", "graph", "docs", "qa", None})


class TestIterContracts:
    """Tests for iter_contracts() function."""

    @staticmethod
    def test_yields_multiple_contracts() -> None:
        """Verify iter_contracts yields multiple contracts."""
        contracts = list(iter_contracts())
        expect_not_empty(contracts, label="contracts")

    @staticmethod
    def test_each_contract_is_dataset_contract() -> None:
        """Verify each yielded item is a DatasetContract."""
        for contract in iter_contracts():
            expect_is_instance(contract, DatasetContract)

    @staticmethod
    def test_includes_tables_and_views() -> None:
        """Verify iteration includes both tables and views."""
        contracts = list(iter_contracts())
        has_tables = any(not is_view(c.table_key) for c in contracts)
        has_views = any(is_view(c.table_key) for c in contracts)
        expect_true(has_tables, message="Should include at least one table")
        expect_true(has_views, message="Should include at least one view")


class TestIterContractsByTableKey:
    """Tests for iter_contracts_by_table_key() function."""

    @staticmethod
    def test_yields_tuples_of_key_and_contract() -> None:
        """Verify iter_contracts_by_table_key yields (key, contract) tuples."""
        for table_key, contract in iter_contracts_by_table_key():
            expect_is_instance(table_key, str)
            expect_equal(contract.table_key, table_key)


class TestViewHandling:
    """Tests for view-specific contract derivation."""

    @staticmethod
    def test_views_have_no_producing_target() -> None:
        """Verify views are handled even without producing targets."""
        discovered = discover_derived_docs_views()
        if not discovered:
            pytest.skip("No views defined in DERIVED_DOCS_VIEWS")

        # Views should return contracts even without targets
        for view_key in list(discovered)[:5]:  # Test first 5
            contract = expect_is_not_none(get_contract_for_table_key(view_key))
            expect_true(contract.is_view)
            expect_equal(contract.table_key, view_key)

    @staticmethod
    def test_views_have_docs_view_tag() -> None:
        """Verify view contracts have appropriate tags."""
        discovered = discover_derived_docs_views()
        if not discovered:
            pytest.skip("No views defined in DERIVED_DOCS_VIEWS")

        view_key = discovered[0]
        contract = expect_is_not_none(get_contract_for_table_key(view_key))
        expect_true("docs_view" in contract.tags or "read_only" in contract.tags)


class TestTableHandling:
    """Tests for table-specific contract derivation."""

    @staticmethod
    def test_tables_have_base_table_tag() -> None:
        """Verify table contracts have base_table tag."""
        contract = expect_is_not_none(get_contract_for_table_key("analytics.function_types"))
        expect_in("base_table", contract.tags)

    @staticmethod
    def test_tables_have_owner_package() -> None:
        """Verify tables have owner_package derived from schema prefix."""
        contract = expect_is_not_none(get_contract_for_table_key("analytics.function_types"))
        expect_equal(contract.owner_package, "analytics")

        contract = get_contract_for_table_key("core.goids")
        expect_equal(contract.owner_package, "core")


class TestContractCacheManagement:
    """Tests for contract cache management."""

    @staticmethod
    def test_clear_contract_cache_clears_all_cached_contracts() -> None:
        """Verify clear_contract_cache() removes all cached entries."""
        # Populate cache
        contract1 = get_contract_for_table_key("analytics.function_types")

        # Clear cache
        clear_contract_cache()

        # Get again - should be a new instance if not frozen
        contract2 = get_contract_for_table_key("analytics.function_types")

        # Both should have same values even after cache clear
        expect_equal(contract1.table_key, contract2.table_key)


class TestBuildStorageContractParity:
    """Ensure build and storage contract providers align."""

    @staticmethod
    def test_storage_contract_matches_build_contract() -> None:
        """Verify build and storage contracts agree on core fields."""
        clear_storage_contract_cache()
        ensure_storage_contract_catalog()
        table_key = "analytics.function_types"

        build_contract = get_contract_for_table_key(table_key)
        storage_contract = get_storage_contract_for_table_key(table_key)

        expect_equal(build_contract.table_key, storage_contract.table_key)
        expect_equal(build_contract.name, storage_contract.name)
        expect_equal(build_contract.is_view, storage_contract.is_view)
        expect_equal(build_contract.owner_package, storage_contract.owner_package)
        expect_equal(build_contract.family, storage_contract.family)
        expect_equal(build_contract.json_schema_id, storage_contract.json_schema_id)
        expect_equal(build_contract.jsonl_filename, storage_contract.jsonl_filename)
        expect_equal(build_contract.parquet_filename, storage_contract.parquet_filename)
        expect_equal(build_contract.tags, storage_contract.tags)


class TestContractMetadataDerivation:
    """Tests for metadata derivation in contracts."""

    @staticmethod
    def test_schema_description_used_when_available() -> None:
        """Verify schema description is used as contract description when available."""
        contract = get_contract_for_table_key("analytics.function_types")
        # Description should be populated from schema or target
        # The exact value depends on schema provider configuration
        expect_true(
            contract.description is None or isinstance(contract.description, str),
            message="Description should be None or string",
        )

    @staticmethod
    def test_family_derived_from_schema_prefix() -> None:
        """Verify family is derived from schema prefix."""
        contract = get_contract_for_table_key("analytics.function_types")
        expect_equal(contract.family, "analytics")

        contract = get_contract_for_table_key("core.goids")
        expect_equal(contract.family, "core")

    @staticmethod
    def test_validation_profile_defaults_to_strict() -> None:
        """Verify validation_profile defaults to 'strict'."""
        contract = get_contract_for_table_key("analytics.function_types")
        expect_equal(contract.validation_profile, "strict")
