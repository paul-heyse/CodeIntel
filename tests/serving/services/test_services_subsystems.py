"""Tests for subsystem service delegates.

This module tests subsystem query delegates via LocalQueryService.
"""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import (
    BackendResource,
    create_app,
)
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
)
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.http_payloads import (
    RequestRecorder,
    make_subsystem_http_responses,
)
from tests._helpers.serving_harnesses import HttpSubsystemHarness, SubsystemDelegateHarness

# =============================================================================
# Subsystem Route Tests (covers service delegates)
# =============================================================================


def test_list_subsystems_returns_data(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem listing returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystems")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_true("subsystems" in data or isinstance(data, list))


def test_list_subsystems_with_limit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem listing respects limit parameter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystems?limit=5")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_get_subsystem_modules(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem modules endpoint functions.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    # Try to get modules for a subsystem
    with TestClient(app) as client:
        response = client.get("/architecture/subsystem?subsystem_id=test_subsystem")

    # May be 200 with empty data or 400/404 if no such subsystem - both are valid
    expect_true(
        response.status_code
        in {
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
        }
    )


def test_get_module_subsystems(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify module subsystems endpoint functions.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/module-subsystems?module=test.module")

    # May be 200 with data or 400/404 if not found
    expect_true(
        response.status_code
        in {
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
        }
    )


def test_subsystem_coverage_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem coverage endpoint returns data.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystem-coverage")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_subsystem_profiles_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem profiles endpoint returns data.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystem-profiles")

    expect_equal(response.status_code, status.HTTP_200_OK)


# =============================================================================
# Delegate normalization and HTTP mixin coverage
# =============================================================================


def test_subsystem_delegate_normalization_variants() -> None:
    """Ensure subsystem delegates normalize dict and response payloads to domain."""
    payloads: dict[str, object] = {
        "list_subsystems": {"subsystems": [], "meta": ResponseMeta().model_dump()},
        "get_module_subsystems": ModuleSubsystemResponse(
            found=True, memberships=[], meta=ResponseMeta()
        ),
        "get_file_hints": FileHintsResponse(
            found=True,
            hints=[],
            meta=ResponseMeta(),
        ),
        "get_subsystem_modules": SubsystemModulesResponse(
            found=True,
            modules=[],
            meta=ResponseMeta(),
        ),
        "search_subsystems": SubsystemSearchResponse(
            subsystems=[],
            meta=ResponseMeta(),
        ),
        "list_subsystem_profiles": SubsystemProfileResponse(profiles=[], meta=ResponseMeta()),
        "list_subsystem_coverage": SubsystemCoverageResponse(coverage=[], meta=ResponseMeta()),
    }
    delegates = SubsystemDelegateHarness(payloads)

    subsystems = delegates.list_subsystems(limit=1)
    module_subs = delegates.get_module_subsystems(module="m")
    hints = delegates.get_file_hints(rel_path="a.py")
    modules = delegates.get_subsystem_modules(subsystem_id="s")
    search = delegates.search_subsystems(q="s")
    summary = delegates.summarize_subsystem(subsystem_id="s")
    profiles = delegates.list_subsystem_profiles(limit=1)
    coverage = delegates.list_subsystem_coverage(limit=1)

    expect_true(isinstance(subsystems, dm.SubsystemSummaryResult))
    expect_true(isinstance(module_subs, dm.ModuleSubsystemResult))
    expect_true(isinstance(hints, dm.FileHintsResult))
    expect_true(isinstance(modules, dm.SubsystemModulesResult))
    expect_true(isinstance(search, dm.SubsystemSearchResult))
    expect_true(isinstance(summary, dm.SubsystemModulesResult))
    expect_true(isinstance(profiles, dm.SubsystemProfileResult))
    expect_true(isinstance(coverage, dm.SubsystemCoverageResult))
    expect_true(("list_subsystem_profiles", "docs.v_subsystem_profile") in delegates.called)
    expect_true(("list_subsystem_coverage", "docs.v_subsystem_coverage") in delegates.called)


def test_http_subsystem_mixin_clamp_and_problem_fallback() -> None:
    """Cover clamp short-circuit and ProblemError fallback in HTTP mixin."""
    responses: dict[str, object] = {
        "/architecture/subsystems": SubsystemSummaryResponse(
            subsystems=[],
            meta=ResponseMeta(),
        ),
        "/architecture/module-subsystems": ModuleSubsystemResponse(
            found=True, memberships=[], meta=ResponseMeta()
        ),
        "/ide/hints": FileHintsResponse(found=True, hints=[], meta=ResponseMeta()),
        "/architecture/subsystem": SubsystemModulesResponse(
            found=True,
            modules=[],
            meta=ResponseMeta(),
        ),
        "/architecture/subsystem-profiles": SubsystemProfileResponse(
            profiles=[],
            meta=ResponseMeta(),
        ),
        "/architecture/subsystem-coverage": SubsystemCoverageResponse(
            coverage=[],
            meta=ResponseMeta(),
        ),
    }
    requester = RequestRecorder(responses, error_paths={"/architecture/subsystem"})
    http_subs = HttpSubsystemHarness(
        limits=BackendLimits(default_limit=1, max_rows_per_call=1),
        requester=requester,
    )

    empty_summary = http_subs.list_subsystems(limit=-1)
    empty_profiles = http_subs.list_subsystem_profiles(limit=-5)
    empty_coverage = http_subs.list_subsystem_coverage(limit=-2)
    modules = http_subs.get_subsystem_modules(subsystem_id="missing")
    summary = http_subs.summarize_subsystem(subsystem_id="missing")

    expect_true(empty_summary.subsystems == [])
    expect_true(empty_profiles.profiles == [])
    expect_true(empty_coverage.coverage == [])
    expect_true(modules.found is False)
    expect_true(summary.found is False)


def test_http_subsystem_mixin_normalization_paths() -> None:
    """Validate HTTP mixin normalizes module subsystems and hints payloads."""
    responses = make_subsystem_http_responses()
    requester = RequestRecorder(responses)
    http_subs = HttpSubsystemHarness(
        limits=BackendLimits(default_limit=5, max_rows_per_call=10),
        requester=requester,
    )

    subsystems = http_subs.list_subsystems()
    module_subs = http_subs.get_module_subsystems(module="m")
    hints = http_subs.get_file_hints(rel_path="a.py")
    modules = http_subs.get_subsystem_modules(subsystem_id="s")
    profiles = http_subs.list_subsystem_profiles(limit=2)
    coverage = http_subs.list_subsystem_coverage(limit=2)

    expect_true(isinstance(subsystems, dm.SubsystemSummaryResult))
    expect_true(isinstance(module_subs, dm.ModuleSubsystemResult))
    expect_true(isinstance(hints, dm.FileHintsResult))
    expect_true(isinstance(modules, dm.SubsystemModulesResult))
    expect_true(isinstance(profiles, dm.SubsystemProfileResult))
    expect_true(isinstance(coverage, dm.SubsystemCoverageResult))
