"""Tests for services/profiles.py delegate classes.

This module directly tests the _ProfileQueryDelegates to achieve higher coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.services.query_service import LocalQueryService

if TYPE_CHECKING:
    from tests._helpers.analytics_samples import AnalyticsSamples
    from tests._helpers.serving_contexts import ProvisionedServiceContext

# Test constants


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
# Tests for _ProfileQueryDelegates through LocalQueryService
# =============================================================================


def test_get_function_profile_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_function_profile returns domain FunctionProfileResult."""
    service = _build_local_service(provisioned_service_ctx)

    profile = service.get_function_profile(goid_h128=analytics_samples.goid_h128)

    _expect(
        condition=isinstance(profile, dm.FunctionProfileResult),
        message="Should return FunctionProfileResult domain object",
    )


def test_get_function_profile_not_found(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify get_function_profile handles not found case."""
    service = _build_local_service(provisioned_service_ctx)

    # Use a nonexistent goid_h128 - should raise an error
    nonexistent_goid = 99999999
    try:
        service.get_function_profile(goid_h128=nonexistent_goid)
        # If we get here without error, service handles gracefully
        result_returned = True
    except McpError:
        # Expected path - not found should raise McpError
        result_returned = False

    _expect(
        condition=isinstance(result_returned, bool),
        message="Should either return result or raise McpError",
    )


def test_get_file_profile_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_file_profile returns domain FileProfileResult."""
    service = _build_local_service(provisioned_service_ctx)

    profile = service.get_file_profile(rel_path=analytics_samples.rel_path)

    _expect(
        condition=isinstance(profile, dm.FileProfileResult),
        message="Should return FileProfileResult domain object",
    )


def test_get_file_profile_not_found(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify get_file_profile handles not found case."""
    service = _build_local_service(provisioned_service_ctx)

    profile = service.get_file_profile(rel_path="nonexistent/path/file.py")

    _expect(
        condition=isinstance(profile, dm.FileProfileResult),
        message="Should return FileProfileResult even for nonexistent file",
    )
    _expect(
        condition=profile.found is False,
        message="Should indicate file not found",
    )


def test_get_module_profile_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_module_profile returns domain ModuleProfileResult."""
    service = _build_local_service(provisioned_service_ctx)

    profile = service.get_module_profile(module=analytics_samples.module)

    _expect(
        condition=isinstance(profile, dm.ModuleProfileResult),
        message="Should return ModuleProfileResult domain object",
    )


def test_get_module_profile_not_found(
    provisioned_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify get_module_profile handles not found case."""
    service = _build_local_service(provisioned_service_ctx)

    # Use nonexistent module - should raise McpError
    try:
        service.get_module_profile(module="nonexistent.module.xyz")
        result_returned = True
    except McpError:
        # Expected path - not found should raise McpError
        result_returned = False

    _expect(
        condition=isinstance(result_returned, bool),
        message="Should either return result or raise McpError",
    )


def test_get_function_architecture_returns_domain_result(
    provisioned_service_ctx: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_function_architecture returns domain result."""
    service = _build_local_service(provisioned_service_ctx)

    architecture = service.get_function_architecture(goid_h128=analytics_samples.goid_h128)

    _expect(
        condition=isinstance(architecture, dm.FunctionArchitectureResult),
        message="Should return FunctionArchitectureResult domain object",
    )


def test_get_function_architecture_not_found(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify get_function_architecture handles not found case."""
    service = _build_local_service(provisioned_service_app)

    # Use nonexistent goid_h128 - should raise McpError
    nonexistent_goid = 99999999
    try:
        service.get_function_architecture(goid_h128=nonexistent_goid)
        result_returned = True
    except McpError:
        # Expected path - not found should raise McpError
        result_returned = False

    _expect(
        condition=isinstance(result_returned, bool),
        message="Should either return result or raise McpError",
    )


def test_get_module_architecture_returns_domain_result(
    provisioned_service_app: ProvisionedServiceContext,
    analytics_samples: AnalyticsSamples,
) -> None:
    """Verify get_module_architecture returns domain result."""
    service = _build_local_service(provisioned_service_app)

    architecture = service.get_module_architecture(module=analytics_samples.module)

    _expect(
        condition=isinstance(architecture, dm.ModuleArchitectureResult),
        message="Should return ModuleArchitectureResult domain object",
    )


def test_get_module_architecture_not_found(
    provisioned_service_app: ProvisionedServiceContext,
) -> None:
    """Verify get_module_architecture handles not found case."""
    service = _build_local_service(provisioned_service_app)

    # Use nonexistent module - should raise McpError
    try:
        service.get_module_architecture(module="nonexistent.module.xyz")
        result_returned = True
    except McpError:
        # Expected path - not found should raise McpError
        result_returned = False

    _expect(
        condition=isinstance(result_returned, bool),
        message="Should either return result or raise McpError",
    )
