"""Comprehensive tests for MCP backend implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from pydantic import ValidationError

from codeintel.config.serving_models import ServingConfig
from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.bootstrap import build_backend_resource
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.mcp.models import (
    DatasetSpecDescriptor,
    FunctionSummaryResponse,
    ResponseMeta,
)
from codeintel.serving.services.errors import DatasetNotFoundError, ProblemDetail, ProblemError
from codeintel.serving.services.query_service import HttpQueryService, LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.analytics_samples import AnalyticsSamples, architecture_seed_selector
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.dataset_factories import make_descriptor, sample_dataset_specs
from tests._helpers.fakes.serving_backends import build_serving_backend
from tests._helpers.http_backend import HttpBackendTestConfig, make_http_backend_with_responses
from tests._helpers.http_payloads import make_problem_detail_payload, make_retry_sequence
from tests._helpers.plugins.mcp import McpBackendComponents
from tests._helpers.serving_stubs import HookedDuckDBQueryApi

if TYPE_CHECKING:
    from tests._helpers.context import TestContext

# Test constants
CUSTOM_DEFAULT_LIMIT = 25
CUSTOM_MAX_ROWS = 250


def _build_arch_backend(
    architecture_gateway: StorageGateway,
    factory: Callable[..., McpBackendComponents],
) -> DuckDBBackend:
    """Build backend components for the seeded architecture gateway.

    Returns
    -------
    DuckDBBackend
        Backend bound to the architecture gateway snapshot.
    """
    return factory(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    ).backend


@pytest.fixture
def architecture_samples(
    architecture_gateway: StorageGateway,
) -> AnalyticsSamples:
    """Analytics identifiers for the seeded architecture gateway.

    Returns
    -------
    AnalyticsSamples
        Sample identifiers loaded from the architecture gateway.
    """
    return architecture_seed_selector(architecture_gateway)


# =============================================================================
# DuckDBBackend Construction Tests
# =============================================================================


def test_duckdb_backend_creation(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify DuckDBBackend can be constructed with provisioned gateway."""
    backend = mcp_backend_components.backend

    expect_true(backend.gateway is mcp_backend_components.gateway)
    expect_equal(backend.repo, mcp_backend_components.repo)
    expect_equal(backend.commit, mcp_backend_components.commit)


def test_duckdb_backend_with_custom_limits(
    provisioned_repo: TestContext,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify DuckDBBackend respects custom limits."""
    custom_limits = BackendLimits(
        default_limit=CUSTOM_DEFAULT_LIMIT, max_rows_per_call=CUSTOM_MAX_ROWS
    )
    backend = mcp_backend_factory(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=custom_limits,
    ).backend

    expect_equal(backend.limits.default_limit, CUSTOM_DEFAULT_LIMIT)
    expect_equal(backend.limits.max_rows_per_call, CUSTOM_MAX_ROWS)


def test_duckdb_backend_with_service(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify DuckDBBackend accepts a service parameter."""
    service = LocalQueryService(
        query=mcp_backend_components.query,
        dataset_tables=dict(mcp_backend_components.gateway.datasets.mapping),
    )

    backend = DuckDBBackend(
        service=service,
        gateway=mcp_backend_components.gateway,
        repo=mcp_backend_components.repo,
        commit=mcp_backend_components.commit,
    )

    expect_true(backend.service is service)


# =============================================================================
# Dataset Operations Tests
# =============================================================================


def test_duckdb_backend_list_datasets(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify list_datasets returns dataset descriptors."""
    datasets = mcp_backend_components.backend.list_datasets()

    expect_is_instance(datasets, list)


def test_duckdb_backend_dataset_specs(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify dataset_specs returns spec descriptors."""
    specs = mcp_backend_components.backend.dataset_specs()

    expect_is_instance(specs, list)


# =============================================================================
# Function Operations Tests
# =============================================================================


def test_duckdb_backend_list_high_risk_functions(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify list_high_risk_functions works with real gateway."""
    response = mcp_backend_components.backend.list_high_risk_functions(min_risk=0.5, limit=10)

    expect_true(hasattr(response, "functions"))


def test_duckdb_backend_list_high_risk_functions_with_tested_only(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify list_high_risk_functions accepts tested_only filter."""
    response = mcp_backend_components.backend.list_high_risk_functions(
        min_risk=0.5, limit=10, tested_only=True
    )

    expect_true(hasattr(response, "functions"))


# =============================================================================
# Subsystem Operations Tests
# =============================================================================


def test_duckdb_backend_list_subsystems(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify list_subsystems works with architecture gateway."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    if architecture_samples.subsystem_id is None:
        pytest.skip("No subsystem data available")

    response = backend.list_subsystems(limit=10)

    expect_true(hasattr(response, "subsystems"))


def test_duckdb_backend_list_subsystems_with_role_filter(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify list_subsystems accepts role filter."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    if architecture_samples.subsystem_id is None:
        pytest.skip("No subsystem data available")

    response = backend.list_subsystems(limit=10, role="test_role")

    expect_true(hasattr(response, "subsystems"))


def test_duckdb_backend_list_subsystems_with_query_filter(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify list_subsystems accepts query filter."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    if architecture_samples.subsystem_id is None:
        pytest.skip("No subsystem data available")

    response = backend.list_subsystems(limit=10, q="test")

    expect_true(hasattr(response, "subsystems"))


def test_duckdb_backend_search_subsystems(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify search_subsystems works with architecture gateway."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    response = backend.search_subsystems(limit=10)

    expect_true(hasattr(response, "results") or hasattr(response, "subsystems"))


# =============================================================================
# Service Access Tests
# =============================================================================


def test_duckdb_backend_service_attribute(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify DuckDBBackend exposes service attribute."""
    expect_is_not_none(mcp_backend_components.backend.service)


def test_duckdb_backend_limits_attribute(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify DuckDBBackend exposes limits attribute."""
    backend = mcp_backend_components.backend

    expect_is_not_none(backend.limits)
    expect_true(hasattr(backend.limits, "default_limit"))
    expect_true(hasattr(backend.limits, "max_rows_per_call"))


# =============================================================================
# Extended Function Operations Tests
# =============================================================================


def test_duckdb_backend_get_function_summary(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_function_summary works with goid_h128."""
    backend = mcp_backend_components.backend

    # Get a valid goid_h128
    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    try:
        response = backend.get_function_summary(goid_h128=goid_h128)
        expect_is_not_none(response)
    except McpError:
        # Expected when function summary is not found
        pass


def test_duckdb_backend_get_callgraph_neighbors(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_callgraph_neighbors works with goid_h128."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="both")
    expect_true(hasattr(response, "incoming") or hasattr(response, "outgoing"))


def test_duckdb_backend_get_callgraph_neighbors_direction_in(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_callgraph_neighbors works with direction=in."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="in")
    expect_is_not_none(response)


def test_duckdb_backend_get_callgraph_neighborhood(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_callgraph_neighborhood works with goid_h128."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_callgraph_neighborhood(goid_h128=goid_h128, radius=1)
    expect_true(hasattr(response, "nodes"))
    expect_true(hasattr(response, "edges"))


def test_duckdb_backend_get_tests_for_function(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_tests_for_function works with goid_h128."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    response = backend.get_tests_for_function(goid_h128=goid_h128)
    expect_true(hasattr(response, "tests"))


def test_duckdb_backend_get_file_summary(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_file_summary works with rel_path."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    try:
        response = backend.get_file_summary(rel_path=rel_path)
        expect_is_not_none(response)
    except McpError:
        # Expected when file summary is not found
        pass


# =============================================================================
# Profile Operations Tests
# =============================================================================


def test_duckdb_backend_get_function_profile(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_function_profile works with goid_h128."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    try:
        response = backend.get_function_profile(goid_h128=goid_h128)
        expect_is_not_none(response)
    except McpError:
        # Expected when profile is not found
        pass


def test_duckdb_backend_get_file_profile(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_file_profile works with rel_path."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    try:
        response = backend.get_file_profile(rel_path=rel_path)
        expect_is_not_none(response)
    except McpError:
        # Expected when profile is not found
        pass


def test_duckdb_backend_get_module_profile(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_module_profile works with module name."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    try:
        response = backend.get_module_profile(module=module)
        expect_is_not_none(response)
    except McpError:
        # Expected when profile is not found
        pass


def test_duckdb_backend_get_function_architecture(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_function_architecture works with goid_h128."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    try:
        response = backend.get_function_architecture(goid_h128=goid_h128)
        expect_is_not_none(response)
    except McpError:
        # Expected when architecture is not found
        pass


def test_duckdb_backend_get_module_architecture(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_module_architecture works with module name."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    try:
        response = backend.get_module_architecture(module=module)
        expect_is_not_none(response)
    except McpError:
        # Expected when architecture is not found
        pass


# =============================================================================
# Extended Subsystem Operations Tests
# =============================================================================


def test_duckdb_backend_get_module_subsystems(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify get_module_subsystems works with architecture gateway."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    result = architecture_gateway.con.execute("SELECT module FROM core.modules LIMIT 1").fetchone()

    if result is None:
        pytest.skip("No modules available in test data")

    module = result[0]

    try:
        response = backend.get_module_subsystems(module=module)
        expect_is_not_none(response)
    except McpError:
        # Expected when subsystems not found
        pass


def test_duckdb_backend_get_file_hints(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify get_file_hints works with architecture gateway."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    result = architecture_gateway.con.execute("SELECT path FROM core.modules LIMIT 1").fetchone()

    if result is None:
        pytest.skip("No modules available in test data")

    rel_path = result[0]

    try:
        response = backend.get_file_hints(rel_path=rel_path)
        expect_is_not_none(response)
    except McpError:
        # Expected when hints not found
        pass


def test_duckdb_backend_get_subsystem_modules(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify get_subsystem_modules works with architecture gateway."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]

    try:
        response = backend.get_subsystem_modules(subsystem_id=subsystem_id)
        expect_is_not_none(response)
    except McpError:
        # Expected when subsystem not found
        pass


def test_duckdb_backend_summarize_subsystem(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify summarize_subsystem works with architecture gateway."""
    backend = _build_arch_backend(architecture_gateway, mcp_backend_factory)

    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No subsystems available in test data")

    subsystem_id = result[0]

    try:
        response = backend.summarize_subsystem(subsystem_id=subsystem_id)
        expect_is_not_none(response)
    except McpError:
        # Expected when subsystem not found
        pass


# =============================================================================
# Direction Validation via Public API Tests
# =============================================================================


def test_callgraph_neighbors_direction_incoming(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_callgraph_neighbors accepts 'incoming' direction."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    # Should not raise - direction is normalized internally
    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="incoming")
    expect_is_not_none(response)


def test_callgraph_neighbors_direction_outgoing(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_callgraph_neighbors accepts 'outgoing' direction."""
    backend = mcp_backend_components.backend

    result = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]
    # Should not raise - direction is normalized internally
    response = backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="outgoing")
    expect_is_not_none(response)


# =============================================================================
# DuckDBBackend Error Handling Tests
# =============================================================================


def test_duckdb_backend_get_function_summary_missing_identifier(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_function_summary raises when no identifier provided."""
    backend = mcp_backend_components.backend

    with pytest.raises(McpError):
        backend.get_function_summary()


def test_duckdb_backend_get_tests_for_function_missing_identifier(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_tests_for_function raises when no identifier provided."""
    backend = mcp_backend_components.backend

    with pytest.raises(McpError):
        backend.get_tests_for_function()


def test_duckdb_backend_get_callgraph_neighbors_invalid_direction(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_callgraph_neighbors raises for invalid direction."""
    backend = mcp_backend_components.backend

    goid_h128 = 123456
    with pytest.raises(McpError):
        backend.get_callgraph_neighbors(goid_h128=goid_h128, direction="invalid")


def test_duckdb_backend_get_import_boundary(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify get_import_boundary returns response."""
    backend = mcp_backend_components.backend

    # Test with nonexistent subsystem - should return empty boundary
    response = backend.get_import_boundary(subsystem_id="nonexistent_subsystem")
    expect_is_not_none(response)


def test_duckdb_backend_read_dataset_rows(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify read_dataset_rows works for valid datasets."""
    backend = mcp_backend_components.backend

    datasets = backend.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name
    response = backend.read_dataset_rows(dataset_name=dataset_name, limit=5)
    expect_is_not_none(response)


def test_duckdb_backend_read_dataset_rows_nonexistent(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify read_dataset_rows raises for nonexistent dataset."""
    backend = mcp_backend_components.backend

    with pytest.raises(McpError):
        backend.read_dataset_rows(dataset_name="nonexistent_dataset_xyz")


def test_duckdb_backend_dataset_schema(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify dataset_schema works for valid datasets."""
    backend = mcp_backend_components.backend

    datasets = backend.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name
    response = backend.dataset_schema(dataset_name=dataset_name, sample_limit=3)
    expect_is_not_none(response)


def test_duckdb_backend_dataset_schema_nonexistent(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify dataset_schema raises for nonexistent dataset."""
    backend = mcp_backend_components.backend

    with pytest.raises(McpError):
        backend.dataset_schema(dataset_name="nonexistent_dataset_xyz")


# =============================================================================
# Build Backend Resource Factory Tests
# =============================================================================


def test_build_backend_resource_local_db_mode(
    provisioned_repo: TestContext,
) -> None:
    """Verify build_backend_resource creates DuckDBBackend in local_db mode."""
    cfg = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    resource = build_backend_resource(cfg, gateway=provisioned_repo.gateway)

    expect_is_instance(resource.backend, DuckDBBackend)


def test_serving_config_remote_api_missing_url_raises(
    provisioned_repo: TestContext,
) -> None:
    """Verify ServingConfig raises when api_base_url missing in remote mode."""
    with pytest.raises(ValidationError, match="api_base_url is required"):
        ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url=None,
        )


# =============================================================================
# DatasetBackendMixin normalization and error handling
# =============================================================================


class _DatasetService(LocalQueryService):
    """LocalQueryService wrapper providing dataset-only hooks."""

    def __init__(self) -> None:
        self._rows_fail = False
        query = HookedDuckDBQueryApi(
            hooks={
                "dataset_hooks": {
                    "list_datasets": self._list_datasets,
                    "dataset_specs": self._dataset_specs,
                    "dataset_schema": self._dataset_schema,
                    "read_dataset_rows": self._read_dataset_rows,
                },
                "profile_hooks": {"get_file_hints": self._file_hints},
            }
        )
        super().__init__(
            query=query,
            dataset_tables={
                "docs.functions": "docs.v_functions",
                "docs.functions_model": "docs.v_functions_model",
                "docs.functions_payload": "docs.v_functions_payload",
            },
            observability=None,
        )

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        """Return dataset descriptors for the hooked dataset service.

        Returns
        -------
        list[dm.DatasetDescriptorDomain]
            Descriptor rows produced by the dataset hook.
        """
        return cast(
            "list[dm.DatasetDescriptorDomain]",
            self._call("list_datasets", self._list_datasets),
        )

    @staticmethod
    def _list_datasets() -> list[object]:
        domain, model, payload, _meta = make_descriptor(
            name="docs.functions",
            table="docs.v_functions",
            description="fn docs",
        )
        return [domain, model, payload]

    @staticmethod
    def _dataset_specs() -> list[object]:
        specs = sample_dataset_specs()
        return [*specs, *(spec.model_dump() for spec in specs)]

    @staticmethod
    def _dataset_schema(*, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        meta = dm.ResponseMeta(
            requested_limit=sample_limit,
            applied_limit=sample_limit,
            requested_offset=0,
            applied_offset=0,
            truncated=False,
        )
        return dm.DatasetSchema(
            dataset_name=dataset_name,
            table_key="docs.v_functions",
            duckdb_schema=[{"name": "goid", "type": "BIGINT", "nullable": False}],
            json_schema={"type": "object"},
            sample_rows=[{"goid": 1, "name": "fn"}],
            capabilities={"validation": True},
            owner="analytics",
            freshness_sla=None,
            retention_policy=None,
            schema_version="1",
            stable_id="stable-1",
            validation_profile="strict",
            meta=meta,
        )

    def _read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        if self._rows_fail:
            detail = ProblemDetail(
                type="error",
                title="fail",
                detail="bad rows",
                status=400,
                code="bad",
            )
            raise DatasetNotFoundError(detail)
        applied_limit = limit or BackendLimits().default_limit
        meta = dm.ResponseMeta(
            requested_limit=limit,
            applied_limit=applied_limit,
            requested_offset=offset,
            applied_offset=offset,
            truncated=False,
        )
        return dm.DatasetRows(
            dataset_name=dataset_name,
            limit=applied_limit,
            offset=offset,
            rows=[{"goid": 1, "name": "fn"}, {"goid": 2, "name": "fn2"}],
            meta=meta,
        )

    def enable_rows_failure(self) -> None:
        self._rows_fail = True

    @staticmethod
    def _file_hints(*, rel_path: str) -> dm.FileHintsResult:
        _ = rel_path
        return dm.FileHintsResult(found=False, hints=[], meta=dm.ResponseMeta())

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        return cast(
            "list[DatasetSpecDescriptor]",
            self._call("dataset_specs", self._dataset_specs),
        )

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        return self._call(
            "dataset_schema",
            lambda: self._dataset_schema(dataset_name=dataset_name, sample_limit=sample_limit),
            dataset=dataset_name,
            schema_version="1",
        )

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        return self._call(
            "read_dataset_rows",
            lambda: self._read_dataset_rows(dataset_name=dataset_name, limit=limit, offset=offset),
            dataset=dataset_name,
        )


def test_dataset_backend_normalization_variants() -> None:
    """Cover DatasetBackendMixin normalization for dict, dataclass, and model_dump."""
    service = _DatasetService()
    backend_handle = build_serving_backend(service=service)
    backend = DuckDBBackend(
        service=service,
        gateway=backend_handle.backend.gateway,
        repo="demo/repo",
        commit="deadbeef",
    )

    descriptors = backend.list_datasets()
    expect_equal(len(descriptors), 3)
    expect_true(all(isinstance(item.description, str) for item in descriptors))

    specs = backend.dataset_specs()
    expect_is_instance(specs, list)

    schema = backend.dataset_schema(dataset_name="docs.functions")
    schema_meta = schema.meta
    expect_true(schema_meta is not None)
    if schema_meta is not None:
        schema_limit = schema_meta.applied_limit
        expect_true(schema_limit is not None)
        if schema_limit is not None:
            expect_is_instance(schema_limit, int)

    rows = backend.read_dataset_rows(dataset_name="docs.functions", limit=1, offset=0)
    rows_meta = rows.meta
    expect_true(rows_meta is not None)
    if rows_meta is not None:
        rows_limit = rows_meta.applied_limit
        expect_true(rows_limit is not None)
        if rows_limit is not None:
            expect_equal(rows_limit, 1)

    backend_handle.close()


def test_dataset_backend_rows_error_translated_to_mcp() -> None:
    """Ensure DatasetNotFoundError is translated to McpError."""
    service = _DatasetService()
    service.enable_rows_failure()
    backend_handle = build_serving_backend(service=service)
    backend = DuckDBBackend(
        service=service,
        gateway=backend_handle.backend.gateway,
        repo="demo/repo",
        commit="deadbeef",
    )

    with pytest.raises(McpError):
        _ = backend.read_dataset_rows(dataset_name="missing", limit=1)

    backend_handle.close()


def test_dataset_backend_problem_error_translated() -> None:
    """Ensure ProblemError is translated to McpError for dataset_schema."""

    class _ProblemService(LocalQueryService):
        def __init__(self) -> None:
            query = HookedDuckDBQueryApi(
                hooks={
                    "dataset_hooks": {
                        "dataset_schema": self._dataset_schema,
                        "list_datasets": lambda **_: [],
                        "dataset_specs": lambda **_: [],
                        "read_dataset_rows": lambda **_: dm.DatasetRows(
                            dataset_name="docs.functions",
                            limit=1,
                            offset=0,
                            rows=[],
                            meta=dm.ResponseMeta(),
                        ),
                    }
                }
            )
            super().__init__(
                query=query,
                dataset_tables={"docs.functions": "docs.v_functions"},
                observability=None,
            )

        @staticmethod
        def _dataset_schema(*, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
            _ = (dataset_name, sample_limit)
            detail = ProblemDetail(
                type="about:blank",
                title="oops",
                detail="bad schema",
                status=400,
                code="bad",
            )
            raise ProblemError(detail)

        def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
            return self._dataset_schema(dataset_name=dataset_name, sample_limit=sample_limit)

    service = _ProblemService()
    backend_handle = build_serving_backend(service=service)
    backend = DuckDBBackend(
        service=service,
        gateway=backend_handle.backend.gateway,
        repo="demo/repo",
        commit="deadbeef",
    )

    with pytest.raises(McpError):
        _ = backend.dataset_schema(dataset_name="docs.functions")

    backend_handle.close()


# =============================================================================
# Validation helpers
# =============================================================================


def test_duckdb_backend_identifier_validation(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Ensure identifier validation errors are raised."""
    backend = mcp_backend_components.backend

    with pytest.raises(errors.McpError):
        _ = backend.get_function_summary()

    with pytest.raises(errors.McpError):
        _ = backend.get_callgraph_neighbors(goid_h128=1, direction="sideways")


# =============================================================================
# Domain conversion coverage on real gateway
# =============================================================================


def test_duckdb_backend_domain_conversions(mcp_backend_components: McpBackendComponents) -> None:
    """Exercise DuckDBBackend conversions using real gateway data."""
    backend = mcp_backend_components.backend

    goid_row = mcp_backend_components.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()
    module_row = mcp_backend_components.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()
    file_row = mcp_backend_components.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()
    subsystem_row = mcp_backend_components.gateway.con.execute(
        "SELECT subsystem_id FROM analytics.subsystems LIMIT 1"
    ).fetchone()

    if goid_row is None or module_row is None or file_row is None or subsystem_row is None:
        pytest.skip("Required seed data missing")

    goid_h128 = goid_row[0]
    module_name = module_row[0]
    rel_path = file_row[0]
    subsystem_id = subsystem_row[0]

    expect_is_instance(
        backend.list_high_risk_functions(min_risk=0.5),
        dm.HighRiskFunctionsResult,
    )
    expect_is_instance(
        backend.get_callgraph_neighbors(goid_h128=goid_h128),
        dm.CallGraphNeighbors,
    )
    expect_is_instance(
        backend.get_callgraph_neighborhood(goid_h128=goid_h128, radius=1),
        dm.GraphNeighborhood,
    )
    expect_is_instance(
        backend.get_import_boundary(subsystem_id=subsystem_id),
        dm.ImportBoundary,
    )
    expect_is_instance(
        backend.get_tests_for_function(goid_h128=goid_h128),
        dm.TestsForFunctionResult,
    )
    expect_is_instance(
        backend.get_file_summary(rel_path=rel_path),
        dm.FileSummaryResult,
    )
    expect_is_instance(
        backend.get_function_profile(goid_h128=goid_h128),
        dm.FunctionProfileResult,
    )
    expect_is_instance(
        backend.get_file_profile(rel_path=rel_path),
        dm.FileProfileResult,
    )
    expect_is_instance(
        backend.get_module_profile(module=module_name),
        dm.ModuleProfileResult,
    )
    expect_is_instance(
        backend.get_function_architecture(goid_h128=goid_h128),
        dm.FunctionArchitectureResult,
    )
    expect_is_instance(
        backend.get_module_architecture(module=module_name),
        dm.ModuleArchitectureResult,
    )
    expect_is_instance(
        backend.list_subsystems(),
        dm.SubsystemSummaryResult,
    )
    expect_is_instance(
        backend.get_module_subsystems(module=module_name),
        dm.ModuleSubsystemResult,
    )
    expect_is_instance(
        backend.get_subsystem_modules(subsystem_id=subsystem_id),
        dm.SubsystemModulesResult,
    )
    expect_is_instance(
        backend.search_subsystems(q=subsystem_id),
        dm.SubsystemSearchResult,
    )
    expect_is_instance(
        backend.summarize_subsystem(subsystem_id=subsystem_id),
        dm.SubsystemModulesResult,
    )
    expect_is_instance(
        backend.get_file_hints(rel_path=rel_path),
        dm.FileHintsResult,
    )
    expect_is_instance(
        backend.service.list_subsystem_profiles(),
        dm.SubsystemProfileResult,
    )
    expect_is_instance(
        backend.service.list_subsystem_coverage(),
        dm.SubsystemCoverageResult,
    )


# =============================================================================
# HttpBackend coverage
# =============================================================================


def test_http_backend_health_and_request_success() -> None:
    """Verify HttpBackend health check and successful JSON request."""
    backend = make_http_backend_with_responses(
        [
            (200, {"ok": True}),
            (200, {"ok": True}),
        ]
    )

    payload = backend.request_json("/health", {})
    expect_true(payload["ok"])


def test_http_backend_retry_and_circuit_breaker(caplog: pytest.LogCaptureFixture) -> None:
    """Cover retry path and circuit-open guard with warning logs."""
    cfg = HttpBackendTestConfig(
        retry_attempts=2,
        backoff=0.0,
        circuit_threshold=1,
        circuit_cooldown_s=100.0,
    )
    backend = make_http_backend_with_responses(make_retry_sequence(), config=cfg)

    payload = backend.request_json("/functions/high-risk", {})
    expect_true(payload["ok"])
    expect_equal(backend.last_retry_attempts, 2)

    circuit_backend = make_http_backend_with_responses(
        make_retry_sequence()[0:2],
        config=HttpBackendTestConfig(
            retry_attempts=1,
            backoff=0.0,
            circuit_threshold=1,
            circuit_cooldown_s=100.0,
        ),
    )

    with caplog.at_level("WARNING"), pytest.raises(McpError):
        _ = circuit_backend.request_json("/functions/high-risk", {})

    assert_logged(caplog.records, level="WARNING", containing="HTTP backend error")


def test_http_backend_problem_detail_response(caplog: pytest.LogCaptureFixture) -> None:
    """Ensure 4xx responses raise McpError with ProblemDetail payload and logs warning."""
    backend = make_http_backend_with_responses(
        [
            (200, {"ok": True}),  # health
            (404, make_problem_detail_payload()),
        ]
    )

    with caplog.at_level("WARNING"), pytest.raises(McpError):
        _ = backend.request_json("/missing", {})
    assert_logged(caplog.records, level="WARNING", containing="HTTP backend error")


def test_http_backend_async_close_when_owned() -> None:
    """Ensure close handles async clients when owned."""
    backend = make_http_backend_with_responses([(200, {"ok": True})])
    async_client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True})),
    )
    backend.set_async_client(async_client)

    backend.close()
    expect_true(async_client.is_closed)


def test_http_backend_service_override_normalization() -> None:
    """Validate service_override is honored and normalization accepts response models."""
    service = HttpQueryService(
        request_json=lambda _path, _params: FunctionSummaryResponse(
            found=True, summary=None, meta=ResponseMeta()
        ).model_dump(),
        limits=BackendLimits(default_limit=5, max_rows_per_call=10),
    )
    backend = make_http_backend_with_responses(
        [(200, {"ok": True})],
        service_override=service,
    )

    response = backend.get_function_summary(goid_h128=1)
    expect_true(response.found)
