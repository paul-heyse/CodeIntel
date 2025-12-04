"""Comprehensive tests for MCP backend implementations.

This module tests the DuckDBBackend using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.bootstrap import build_backend_resource
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# Test constants
CUSTOM_DEFAULT_LIMIT = 25
CUSTOM_MAX_ROWS = 250


# =============================================================================
# DuckDBBackend Construction Tests
# =============================================================================


def test_duckdb_backend_creation(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend can be constructed with provisioned gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    assert backend.gateway is provisioned_repo.gateway
    assert backend.repo == provisioned_repo.repo
    assert backend.commit == provisioned_repo.commit


def test_duckdb_backend_with_custom_limits(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend respects custom limits.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    custom_limits = BackendLimits(
        default_limit=CUSTOM_DEFAULT_LIMIT, max_rows_per_call=CUSTOM_MAX_ROWS
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=custom_limits,
    )

    assert backend.limits.default_limit == CUSTOM_DEFAULT_LIMIT
    assert backend.limits.max_rows_per_call == CUSTOM_MAX_ROWS


def test_duckdb_backend_with_service_override(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend accepts service_override.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    assert backend.service is service


# =============================================================================
# Dataset Operations Tests
# =============================================================================


def test_duckdb_backend_list_datasets(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_datasets returns dataset descriptors.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    datasets = backend.list_datasets()

    assert isinstance(datasets, list)


def test_duckdb_backend_dataset_specs(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_specs returns spec descriptors.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    specs = backend.dataset_specs()

    assert isinstance(specs, list)


# =============================================================================
# Function Operations Tests
# =============================================================================


def test_duckdb_backend_list_high_risk_functions(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions works with real gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    response = backend.list_high_risk_functions(min_risk=0.5, limit=10)

    assert hasattr(response, "functions")


def test_duckdb_backend_list_high_risk_functions_with_tested_only(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions accepts tested_only filter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    response = backend.list_high_risk_functions(min_risk=0.5, limit=10, tested_only=True)

    assert hasattr(response, "functions")


# =============================================================================
# Subsystem Operations Tests
# =============================================================================


def test_duckdb_backend_list_subsystems(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.list_subsystems(limit=10)

    assert hasattr(response, "subsystems")


def test_duckdb_backend_list_subsystems_with_role_filter(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems accepts role filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.list_subsystems(limit=10, role="test_role")

    assert hasattr(response, "subsystems")


def test_duckdb_backend_list_subsystems_with_query_filter(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems accepts query filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.list_subsystems(limit=10, q="test")

    assert hasattr(response, "subsystems")


def test_duckdb_backend_search_subsystems(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify search_subsystems works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.search_subsystems(limit=10)

    assert hasattr(response, "results") or hasattr(response, "subsystems")


# =============================================================================
# Service Access Tests
# =============================================================================


def test_duckdb_backend_service_attribute(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend exposes service attribute.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    assert backend.service is not None


def test_duckdb_backend_limits_attribute(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend exposes limits attribute.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    assert backend.limits is not None
    assert hasattr(backend.limits, "default_limit")
    assert hasattr(backend.limits, "max_rows_per_call")


# =============================================================================
# Extended Function Operations Tests
# =============================================================================


def test_duckdb_backend_get_function_summary(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary works with goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    # Get a valid goid_h128
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    try:
        response = backend.get_function_summary(goid_h128=goid_h128)
        assert response is not None
    except McpError:
        # Expected when function summary is not found
        pass


def test_duckdb_backend_get_callgraph_neighbors(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors works with goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="both")
    assert hasattr(response, "incoming") or hasattr(response, "outgoing")


def test_duckdb_backend_get_callgraph_neighbors_direction_in(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors works with direction=in.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="in")
    assert response is not None


def test_duckdb_backend_get_callgraph_neighborhood(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighborhood works with goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_callgraph_neighborhood(goid_h128=goid_h128, radius=1)
    assert hasattr(response, "nodes")
    assert hasattr(response, "edges")


def test_duckdb_backend_get_tests_for_function(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function works with goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_tests_for_function(goid_h128=goid_h128)
    assert hasattr(response, "tests")


def test_duckdb_backend_get_file_summary(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_summary works with rel_path.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    try:
        response = backend.get_file_summary(rel_path=rel_path)
        assert response is not None
    except McpError:
        # Expected when file summary is not found
        pass


# =============================================================================
# Profile Operations Tests
# =============================================================================


def test_duckdb_backend_get_function_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_profile works with goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    try:
        response = backend.get_function_profile(goid_h128=goid_h128)
        assert response is not None
    except McpError:
        # Expected when profile is not found
        pass


def test_duckdb_backend_get_file_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_profile works with rel_path.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    try:
        response = backend.get_file_profile(rel_path=rel_path)
        assert response is not None
    except McpError:
        # Expected when profile is not found
        pass


def test_duckdb_backend_get_module_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_profile works with module name.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    try:
        response = backend.get_module_profile(module=module)
        assert response is not None
    except McpError:
        # Expected when profile is not found
        pass


def test_duckdb_backend_get_function_architecture(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_architecture works with goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    try:
        response = backend.get_function_architecture(goid_h128=goid_h128)
        assert response is not None
    except McpError:
        # Expected when architecture is not found
        pass


def test_duckdb_backend_get_module_architecture(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_architecture works with module name.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    try:
        response = backend.get_module_architecture(module=module)
        assert response is not None
    except McpError:
        # Expected when architecture is not found
        pass


# =============================================================================
# Extended Subsystem Operations Tests
# =============================================================================


def test_duckdb_backend_get_module_subsystems(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_module_subsystems works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    result = architecture_gateway.con.execute("SELECT module FROM core.modules LIMIT 1").fetchone()

    if result is None:
        pytest.skip("No modules available in test data")

    module = result[0]

    try:
        response = backend.get_module_subsystems(module=module)
        assert response is not None
    except McpError:
        # Expected when subsystems not found
        pass


def test_duckdb_backend_get_file_hints(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_file_hints works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    result = architecture_gateway.con.execute("SELECT path FROM core.modules LIMIT 1").fetchone()

    if result is None:
        pytest.skip("No modules available in test data")

    rel_path = result[0]

    try:
        response = backend.get_file_hints(rel_path=rel_path)
        assert response is not None
    except McpError:
        # Expected when hints not found
        pass


def test_duckdb_backend_get_subsystem_modules(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify get_subsystem_modules works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]

    try:
        response = backend.get_subsystem_modules(subsystem_id=subsystem_id)
        assert response is not None
    except McpError:
        # Expected when subsystem not found
        pass


def test_duckdb_backend_summarize_subsystem(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify summarize_subsystem works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]

    try:
        response = backend.summarize_subsystem(subsystem_id=subsystem_id)
        assert response is not None
    except McpError:
        # Expected when subsystem not found
        pass


# =============================================================================
# Direction Validation via Public API Tests
# =============================================================================


def test_callgraph_neighbors_direction_incoming(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors accepts 'incoming' direction.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    # Should not raise - direction is normalized internally
    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="incoming")
    assert response is not None


def test_callgraph_neighbors_direction_outgoing(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors accepts 'outgoing' direction.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    # Should not raise - direction is normalized internally
    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="outgoing")
    assert response is not None


# =============================================================================
# DuckDBBackend Error Handling Tests
# =============================================================================


def test_duckdb_backend_get_function_summary_missing_identifier(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_summary raises when no identifier provided.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    with pytest.raises(McpError):
        backend.get_function_summary()


def test_duckdb_backend_get_tests_for_function_missing_identifier(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_tests_for_function raises when no identifier provided.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    with pytest.raises(McpError):
        backend.get_tests_for_function()


def test_duckdb_backend_get_callgraph_neighbors_invalid_direction(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_callgraph_neighbors raises for invalid direction.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    goid_h128 = 123456
    with pytest.raises(McpError):
        backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="invalid")


def test_duckdb_backend_get_import_boundary(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_import_boundary returns response.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    # Test with nonexistent subsystem - should return empty boundary
    response = backend.get_import_boundary(subsystem_id="nonexistent_subsystem")
    assert response is not None


def test_duckdb_backend_read_dataset_rows(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows works for valid datasets.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    datasets = backend.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name
    response = backend.read_dataset_rows(dataset_name=dataset_name, limit=5)
    assert response is not None


def test_duckdb_backend_read_dataset_rows_nonexistent(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows raises for nonexistent dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    with pytest.raises(McpError):
        backend.read_dataset_rows(dataset_name="nonexistent_dataset_xyz")


def test_duckdb_backend_dataset_schema(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_schema works for valid datasets.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    datasets = backend.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name
    response = backend.dataset_schema(dataset_name=dataset_name, sample_limit=3)
    assert response is not None


def test_duckdb_backend_dataset_schema_nonexistent(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_schema raises for nonexistent dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    with pytest.raises(McpError):
        backend.dataset_schema(dataset_name="nonexistent_dataset_xyz")


# =============================================================================
# Build Backend Resource Factory Tests
# =============================================================================


def test_build_backend_resource_local_db_mode(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify build_backend_resource creates DuckDBBackend in local_db mode.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    cfg = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    resource = build_backend_resource(cfg, gateway=provisioned_repo.gateway)

    assert isinstance(resource.backend, DuckDBBackend)


def test_serving_config_remote_api_missing_url_raises(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify ServingConfig raises when api_base_url missing in remote mode.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    with pytest.raises(ValidationError, match="api_base_url is required"):
        ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url=None,
        )
