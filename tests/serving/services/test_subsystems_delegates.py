"""Tests for services/subsystems.py delegate classes.

This module directly tests the _SubsystemQueryDelegates to achieve higher coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers.fixtures import ProvisionedGateway

# Test constants
DEFAULT_LIMIT: Final = 10
MAX_ROWS: Final = 100
LIMIT_FIVE: Final = 5


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


def _build_local_service(
    provisioned_repo: ProvisionedGateway,
) -> LocalQueryService:
    """Build a LocalQueryService for direct testing.

    Returns
    -------
    LocalQueryService
        Configured local query service.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    return LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )


# =============================================================================
# Tests for _SubsystemQueryDelegates through LocalQueryService
# =============================================================================


def test_list_subsystems_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystems returns domain SubsystemSummaryResult."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystems()

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_list_subsystems_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystems with limit parameter."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystems(limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_list_subsystems_with_role_filter(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystems with role filter."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystems(role="api")

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_list_subsystems_with_query(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystems with query filter."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystems(q="test")

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_get_module_subsystems_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_subsystems returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM analytics.subsystem_modules LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystem modules available in test data")

    module = result[0]
    subsystems = service.get_module_subsystems(module=module)

    _expect(
        condition=isinstance(subsystems, dm.ModuleSubsystemResult),
        message="Should return ModuleSubsystemResult domain object",
    )


def test_get_module_subsystems_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_subsystems handles not found case."""
    service = _build_local_service(provisioned_repo)

    subsystems = service.get_module_subsystems(module="nonexistent.module.xyz")

    _expect(
        condition=isinstance(subsystems, dm.ModuleSubsystemResult),
        message="Should return ModuleSubsystemResult even for nonexistent module",
    )


def test_get_file_hints_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_hints returns domain FileHintsResult."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python files available in test data")

    rel_path = result[0]
    hints = service.get_file_hints(rel_path=rel_path)

    _expect(
        condition=isinstance(hints, dm.FileHintsResult),
        message="Should return FileHintsResult domain object",
    )


def test_get_file_hints_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_hints handles not found case."""
    service = _build_local_service(provisioned_repo)

    hints = service.get_file_hints(rel_path="nonexistent/path/file.py")

    _expect(
        condition=isinstance(hints, dm.FileHintsResult),
        message="Should return FileHintsResult even for nonexistent file",
    )


def test_get_subsystem_modules_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_subsystem_modules returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]
    modules = service.get_subsystem_modules(subsystem_id=subsystem_id)

    _expect(
        condition=isinstance(modules, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_get_subsystem_modules_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_subsystem_modules with module_limit."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]
    modules = service.get_subsystem_modules(subsystem_id=subsystem_id, module_limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(modules, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_get_subsystem_modules_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_subsystem_modules handles not found case."""
    service = _build_local_service(provisioned_repo)

    modules = service.get_subsystem_modules(subsystem_id="nonexistent_subsystem")

    _expect(
        condition=isinstance(modules, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult even for nonexistent",
    )


def test_search_subsystems_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify search_subsystems returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = service.search_subsystems()

    _expect(
        condition=isinstance(result, dm.SubsystemSearchResult),
        message="Should return SubsystemSearchResult domain object",
    )


def test_search_subsystems_with_query(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify search_subsystems with query filter."""
    service = _build_local_service(provisioned_repo)

    result = service.search_subsystems(q="test")

    _expect(
        condition=isinstance(result, dm.SubsystemSearchResult),
        message="Should return SubsystemSearchResult domain object",
    )


def test_search_subsystems_with_role(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify search_subsystems with role filter."""
    service = _build_local_service(provisioned_repo)

    result = service.search_subsystems(role="api")

    _expect(
        condition=isinstance(result, dm.SubsystemSearchResult),
        message="Should return SubsystemSearchResult domain object",
    )


def test_summarize_subsystem_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify summarize_subsystem returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]
    summary = service.summarize_subsystem(subsystem_id=subsystem_id)

    _expect(
        condition=isinstance(summary, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_summarize_subsystem_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify summarize_subsystem with module_limit."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]
    summary = service.summarize_subsystem(subsystem_id=subsystem_id, module_limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(summary, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_list_subsystem_profiles_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystem_profiles returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystem_profiles()

    _expect(
        condition=isinstance(result, dm.SubsystemProfileResult),
        message="Should return SubsystemProfileResult domain object",
    )


def test_list_subsystem_profiles_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystem_profiles with limit."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystem_profiles(limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(result, dm.SubsystemProfileResult),
        message="Should return SubsystemProfileResult domain object",
    )


def test_list_subsystem_coverage_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystem_coverage returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystem_coverage()

    _expect(
        condition=isinstance(result, dm.SubsystemCoverageResult),
        message="Should return SubsystemCoverageResult domain object",
    )


def test_list_subsystem_coverage_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_subsystem_coverage with limit."""
    service = _build_local_service(provisioned_repo)

    result = service.list_subsystem_coverage(limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(result, dm.SubsystemCoverageResult),
        message="Should return SubsystemCoverageResult domain object",
    )
