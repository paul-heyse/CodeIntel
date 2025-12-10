"""Tests for services/subsystems.py delegate classes.

This module directly tests the _SubsystemQueryDelegates to achieve higher coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.services.query_service import LocalQueryService

if TYPE_CHECKING:
    from tests._helpers.analytics_samples import AnalyticsSamples
    from tests._helpers.serving_contexts import ProvisionedServiceContext

# Test constants
LIMIT_FIVE: Final = 5


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


def _build_local_service(
    service_ctx: ProvisionedServiceContext,
) -> LocalQueryService:
    """Return the shared LocalQueryService from the provisioned service app.

    Returns
    -------
    LocalQueryService
        Service instance wired to the provisioned gateway snapshot.
    """
    return service_ctx.service  # type: ignore[no-any-return]


# =============================================================================
# Tests for _SubsystemQueryDelegates through LocalQueryService
# =============================================================================


def test_list_subsystems_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystems returns domain SubsystemSummaryResult."""
    service = _build_local_service(provisioned_service_ctx)

    result = service.list_subsystems()

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_list_subsystems_with_limit(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystems with limit parameter."""
    service = _build_local_service(provisioned_service_ctx)

    result = service.list_subsystems(limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_list_subsystems_with_role_filter(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystems with role filter."""
    service = _build_local_service(provisioned_service_ctx)

    result = service.list_subsystems(role="api")

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_list_subsystems_with_query(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystems with query filter."""
    service = _build_local_service(provisioned_service_ctx)

    result = service.list_subsystems(q="test")

    _expect(
        condition=isinstance(result, dm.SubsystemSummaryResult),
        message="Should return SubsystemSummaryResult domain object",
    )


def test_get_module_subsystems_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_module_subsystems returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    subsystems = service.get_module_subsystems(module=analytics_samples.module)

    _expect(
        condition=isinstance(subsystems, dm.ModuleSubsystemResult),
        message="Should return ModuleSubsystemResult domain object",
    )


def test_get_module_subsystems_not_found(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify get_module_subsystems handles not found case."""
    service = _build_local_service(provisioned_service_ctx)

    subsystems = service.get_module_subsystems(module="nonexistent.module.xyz")

    _expect(
        condition=isinstance(subsystems, dm.ModuleSubsystemResult),
        message="Should return ModuleSubsystemResult even for nonexistent module",
    )


def test_get_file_hints_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_file_hints returns domain FileHintsResult."""
    service = _build_local_service(provisioned_service_ctx)

    hints = service.get_file_hints(rel_path=analytics_samples.rel_path)

    _expect(
        condition=isinstance(hints, dm.FileHintsResult),
        message="Should return FileHintsResult domain object",
    )


def test_get_file_hints_not_found(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify get_file_hints handles not found case."""
    service = _build_local_service(provisioned_service_ctx)

    hints = service.get_file_hints(rel_path="nonexistent/path/file.py")

    _expect(
        condition=isinstance(hints, dm.FileHintsResult),
        message="Should return FileHintsResult even for nonexistent file",
    )


def test_get_subsystem_modules_returns_domain_result(
    provisioned_service_app: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_subsystem_modules returns domain result."""
    service = _build_local_service(provisioned_service_app)

    modules = service.get_subsystem_modules(subsystem_id=analytics_samples.subsystem_id)

    _expect(
        condition=isinstance(modules, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_get_subsystem_modules_with_limit(
    provisioned_service_app: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_subsystem_modules with module_limit."""
    service = _build_local_service(provisioned_service_app)

    modules = service.get_subsystem_modules(
        subsystem_id=analytics_samples.subsystem_id,
        module_limit=LIMIT_FIVE,
    )

    _expect(
        condition=isinstance(modules, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_get_subsystem_modules_not_found(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify get_subsystem_modules handles not found case."""
    service = _build_local_service(provisioned_service_app)

    modules = service.get_subsystem_modules(subsystem_id="nonexistent_subsystem")

    _expect(
        condition=isinstance(modules, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult even for nonexistent",
    )


def test_search_subsystems_returns_domain_result(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify search_subsystems returns domain result."""
    service = _build_local_service(provisioned_service_app)

    result = service.search_subsystems()

    _expect(
        condition=isinstance(result, dm.SubsystemSearchResult),
        message="Should return SubsystemSearchResult domain object",
    )


def test_search_subsystems_with_query(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify search_subsystems with query filter."""
    service = _build_local_service(provisioned_service_app)

    result = service.search_subsystems(q="test")

    _expect(
        condition=isinstance(result, dm.SubsystemSearchResult),
        message="Should return SubsystemSearchResult domain object",
    )


def test_search_subsystems_with_role(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify search_subsystems with role filter."""
    service = _build_local_service(provisioned_service_app)

    result = service.search_subsystems(role="api")

    _expect(
        condition=isinstance(result, dm.SubsystemSearchResult),
        message="Should return SubsystemSearchResult domain object",
    )


def test_summarize_subsystem_returns_domain_result(
    provisioned_service_app: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify summarize_subsystem returns domain result."""
    service = _build_local_service(provisioned_service_app)

    summary = service.summarize_subsystem(subsystem_id=analytics_samples.subsystem_id)

    _expect(
        condition=isinstance(summary, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_summarize_subsystem_with_limit(
    provisioned_service_app: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify summarize_subsystem with module_limit."""
    service = _build_local_service(provisioned_service_app)

    summary = service.summarize_subsystem(
        subsystem_id=analytics_samples.subsystem_id,
        module_limit=LIMIT_FIVE,
    )

    _expect(
        condition=isinstance(summary, dm.SubsystemModulesResult),
        message="Should return SubsystemModulesResult domain object",
    )


def test_list_subsystem_profiles_returns_domain_result(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystem_profiles returns domain result."""
    service = _build_local_service(provisioned_service_app)

    result = service.list_subsystem_profiles()

    _expect(
        condition=isinstance(result, dm.SubsystemProfileResult),
        message="Should return SubsystemProfileResult domain object",
    )


def test_list_subsystem_profiles_with_limit(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystem_profiles with limit."""
    service = _build_local_service(provisioned_service_app)

    result = service.list_subsystem_profiles(limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(result, dm.SubsystemProfileResult),
        message="Should return SubsystemProfileResult domain object",
    )


def test_list_subsystem_coverage_returns_domain_result(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystem_coverage returns domain result."""
    service = _build_local_service(provisioned_service_app)

    result = service.list_subsystem_coverage()

    _expect(
        condition=isinstance(result, dm.SubsystemCoverageResult),
        message="Should return SubsystemCoverageResult domain object",
    )


def test_list_subsystem_coverage_with_limit(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify list_subsystem_coverage with limit."""
    service = _build_local_service(provisioned_service_app)

    result = service.list_subsystem_coverage(limit=LIMIT_FIVE)

    _expect(
        condition=isinstance(result, dm.SubsystemCoverageResult),
        message="Should return SubsystemCoverageResult domain object",
    )
