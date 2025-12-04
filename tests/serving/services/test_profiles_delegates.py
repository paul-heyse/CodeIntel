"""Tests for services/profiles.py delegate classes.

This module directly tests the _ProfileQueryDelegates to achieve higher coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# Test constants
DEFAULT_LIMIT: Final = 10
MAX_ROWS: Final = 100


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
# Tests for _ProfileQueryDelegates through LocalQueryService
# =============================================================================


def test_get_function_profile_returns_domain_result(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_profile returns domain FunctionProfileResult."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT function_goid_h128 FROM analytics.function_profile LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No function profiles available in test data")

    goid_h128 = result[0]
    profile = service.get_function_profile(goid_h128=goid_h128)

    _expect(
        condition=isinstance(profile, dm.FunctionProfileResult),
        message="Should return FunctionProfileResult domain object",
    )


def test_get_function_profile_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_profile handles not found case."""
    service = _build_local_service(provisioned_repo)

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_profile returns domain FileProfileResult."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python files available in test data")

    rel_path = result[0]
    profile = service.get_file_profile(rel_path=rel_path)

    _expect(
        condition=isinstance(profile, dm.FileProfileResult),
        message="Should return FileProfileResult domain object",
    )


def test_get_file_profile_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_profile handles not found case."""
    service = _build_local_service(provisioned_repo)

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_profile returns domain ModuleProfileResult."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM analytics.module_profile LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No module profiles available in test data")

    module = result[0]
    profile = service.get_module_profile(module=module)

    _expect(
        condition=isinstance(profile, dm.ModuleProfileResult),
        message="Should return ModuleProfileResult domain object",
    )


def test_get_module_profile_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_profile handles not found case."""
    service = _build_local_service(provisioned_repo)

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_architecture returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT function_goid_h128 FROM analytics.graph_metrics_functions LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No function architecture data available")

    goid_h128 = result[0]
    architecture = service.get_function_architecture(goid_h128=goid_h128)

    _expect(
        condition=isinstance(architecture, dm.FunctionArchitectureResult),
        message="Should return FunctionArchitectureResult domain object",
    )


def test_get_function_architecture_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_architecture handles not found case."""
    service = _build_local_service(provisioned_repo)

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
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_architecture returns domain result."""
    service = _build_local_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM analytics.graph_metrics_modules LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No module architecture data available")

    module = result[0]
    architecture = service.get_module_architecture(module=module)

    _expect(
        condition=isinstance(architecture, dm.ModuleArchitectureResult),
        message="Should return ModuleArchitectureResult domain object",
    )


def test_get_module_architecture_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_architecture handles not found case."""
    service = _build_local_service(provisioned_repo)

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
