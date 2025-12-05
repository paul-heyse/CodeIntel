"""Pytest configuration for the CodeIntel test suite."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import duckdb
import pytest
from coverage import Coverage

from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import assert_single_edge
from tests._helpers.configs import (
    CoverageEdgeEnv,
    CoverageSeedConfig,
    GatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    SpanTestEnv,
)
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.gateway import memory_con_with_macros
from tests._helpers.orchestration import (
    compute_coverage_edges,
    create_coverage_edge_env,
    create_span_test_env,
    generate_coverage_artifact,
    generate_span_coverage,
    provision_docs_export_ready,
    provision_graph_ready_repo,
    provision_ingested_repo,
    provisioned_gateway,
)
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds import (
    CORE_PACK,
    COVERAGE_PACK,
    GRAPH_PACK,
    METRICS_PACK,
)
from tests._helpers.seeds.architecture import open_seeded_architecture_gateway


@pytest.fixture
def fresh_gateway(tmp_path: Path) -> Iterator[StorageGateway]:
    """Provide an in-memory gateway with schema and views applied.

    Yields
    ------
    StorageGateway
        Gateway configured with schemas/views; caller must not close.
    """
    with provisioned_gateway(
        tmp_path / "fresh",
        config=ProvisioningConfig(run_ingestion=False),
    ) as ctx:
        yield ctx.gateway


@pytest.fixture
def provisioned_repo(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a repo-root and ingest baseline data via production entry points.

    Yields
    ------
    ProvisionedGateway
        Gateway plus repo root populated with baseline ingestion data.
    """
    with provision_ingested_repo(tmp_path / "repo") as ctx:
        yield ctx


@pytest.fixture
def graph_ready_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a repo with graph metrics seeds for graph-centric tests.

    Yields
    ------
    ProvisionedGateway
        Gateway plus repo context seeded with graph metrics data.
    """
    with provision_graph_ready_repo(tmp_path / "repo") as ctx:
        yield ctx


@pytest.fixture
def docs_export_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway ready for docs export scenarios.

    Yields
    ------
    ProvisionedGateway
        Gateway populated with docs export seeds.
    """
    ctx = provision_docs_export_ready(tmp_path, repo="demo/repo", commit="deadbeef")
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def ingestion_only_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway without ingestion for custom seeding.

    Yields
    ------
    ProvisionedGateway
        Gateway prepared with schemas but without ingestion.
    """
    with provisioned_gateway(
        tmp_path / "repo",
        config=ProvisioningConfig(run_ingestion=False),
    ) as ctx:
        yield ctx


@pytest.fixture
def loose_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Opt-out gateway for tests that intentionally drift schemas.

    Yields
    ------
    ProvisionedGateway
        Gateway configured without strict schema enforcement.
    """
    with provisioned_gateway(
        tmp_path / "repo",
        config=ProvisioningConfig(
            run_ingestion=False, gateway_options=GatewayOptions(strict_schema=False)
        ),
    ) as ctx:
        yield ctx


@pytest.fixture
def architecture_gateway(tmp_path: Path) -> Iterator[StorageGateway]:
    """Provide a gateway seeded with architecture data (subsystems, call/import graphs).

    Yields
    ------
    StorageGateway
    Gateway configured with architecture dataset seeds.
    """
    gateway = open_seeded_architecture_gateway(
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / "arch.duckdb",
        strict_schema=True,
    )
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def codeintel_env() -> Iterator[None]:
    """Snapshot CODEINTEL_* environment variables and restore after the test."""
    prior = {key: value for key, value in os.environ.items() if key.startswith("CODEINTEL_")}
    try:
        yield
    finally:
        for key in list(os.environ.keys()):
            if key.startswith("CODEINTEL_") and key not in prior:
                os.environ.pop(key, None)
        for key, value in prior.items():
            os.environ[key] = value


@pytest.fixture
def span_env(tmp_path: Path, fresh_gateway: StorageGateway) -> Iterator[SpanTestEnv]:
    """
    Provide a reusable graph span test environment.

    Yields
    ------
    SpanTestEnv
        Span test environment with seeded modules and GOIDs.
    """
    env = create_span_test_env(tmp_path, fresh_gateway)
    try:
        yield env
    finally:
        env.gateway.close()


@pytest.fixture
def coverage_env(tmp_path: Path) -> Iterator[CoverageEdgeEnv]:
    """
    Provide a coverage edge environment with seeded GOIDs and catalog rows.

    Yields
    ------
    CoverageEdgeEnv
        Coverage environment ready for analytics coverage edge tests.
    """
    env = create_coverage_edge_env(tmp_path)
    try:
        yield env
    finally:
        env.gateway.close()


@pytest.fixture
def coverage_artifact(coverage_env: CoverageEdgeEnv, tmp_path: Path) -> Path:
    """
    Generate a coverage artifact for the seeded coverage environment.

    Returns
    -------
    Path
        Path to the generated coverage data file.
    """
    artifact = generate_coverage_artifact(coverage_env, coverage_file=tmp_path / ".coverage")
    return artifact.coverage_file


@pytest.fixture
def coverage_loader(coverage_artifact: Path) -> Callable[[object], Coverage]:
    """
    Provide a coverage loader callable for tests consuming coverage fixtures.

    Returns
    -------
    Callable[[object], Coverage]
        Loader that returns Coverage loaded from the artifact path.
    """

    def _loader(_cfg: object) -> Coverage:
        cov = Coverage(data_file=str(coverage_artifact))
        cov.load()
        return cov

    return _loader


@pytest.fixture
def coverage_edges_seed(coverage_env: CoverageEdgeEnv, coverage_artifact: Path) -> CoverageEdgeEnv:
    """
    Seed analytics.test_coverage_edges using the shared coverage environment.

    Returns
    -------
    CoverageEdgeEnv
        Coverage environment after edges have been computed.
    """
    compute_coverage_edges(coverage_env, coverage_file=coverage_artifact)
    assert_single_edge(coverage_env.gateway.con)
    return coverage_env


@pytest.fixture
def coverage_env_factory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[CoverageSeedConfig], CoverageEdgeEnv]]:
    """
    Create coverage environments with custom seeds.

    Yields
    ------
    Callable[[CoverageSeedConfig], CoverageEdgeEnv]
        Factory that returns a seeded CoverageEdgeEnv for the given seed.
    """
    envs: list[CoverageEdgeEnv] = []

    def _factory(seed: CoverageSeedConfig) -> CoverageEdgeEnv:
        env = create_coverage_edge_env(tmp_path_factory.mktemp("cov_env"), seed=seed)
        envs.append(env)
        return env

    try:
        yield _factory
    finally:
        for env in envs:
            env.gateway.close()


@pytest.fixture
def coverage_profiles_conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Provide an in-memory DuckDB connection with coverage-related tables.

    Yields
    ------
    duckdb.DuckDBPyConnection
        Connection seeded with schemas for coverage profile tests.
    """
    con = memory_con_with_macros()
    con.execute("CREATE SCHEMA analytics")
    con.execute("CREATE SCHEMA core")
    con.execute(
        """
        CREATE TABLE analytics.test_coverage_edges (
            test_id VARCHAR,
            function_goid_h128 DECIMAL(38,0),
            module VARCHAR,
            covered_lines INTEGER,
            executable_lines INTEGER,
            repo VARCHAR,
            commit VARCHAR,
            rel_path VARCHAR,
            qualname VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_catalog (
            test_id VARCHAR,
            repo VARCHAR,
            commit VARCHAR,
            status VARCHAR,
            duration_ms DOUBLE,
            flaky BOOLEAN
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.subsystem_modules (
            module VARCHAR,
            subsystem_id VARCHAR,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.subsystems (
            subsystem_id VARCHAR,
            name VARCHAR,
            max_risk_score DOUBLE,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_graph_metrics_tests (
            test_id VARCHAR,
            degree INTEGER,
            weighted_degree DOUBLE,
            proj_degree INTEGER,
            proj_weight DOUBLE,
            proj_clustering DOUBLE,
            proj_betweenness DOUBLE,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE core.goids (
            goid_h128 DECIMAL(38,0),
            urn VARCHAR,
            repo VARCHAR,
            commit VARCHAR,
            rel_path VARCHAR,
            qualname VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE core.modules (
            module VARCHAR,
            path VARCHAR,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def span_coverage_artifact(span_env: SpanTestEnv) -> Path:
    """Generate a coverage artifact for the span alignment environment.

    Returns
    -------
    Path
        Path to the generated coverage data file.
    """
    artifact = generate_span_coverage(span_env.repo_root)
    return artifact.coverage_file


# =============================================================================
# Hexagonal Architecture Fixtures
# =============================================================================


@pytest.fixture
def test_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a minimal TestContext ready for seed packs.

    This is the foundational fixture for the new hexagonal test architecture.
    Use `ctx.require(PACK)` to apply seed packs.

    Yields
    ------
    TestContext
        Minimal context with gateway, ready for seeds.
    """
    ctx = create_test_context(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def core_ctx(test_ctx: TestContext) -> TestContext:
    """Provide TestContext with CORE_PACK applied.

    Seeds repo_map, modules, and goids tables with standard test data.

    Returns
    -------
    TestContext
        Context with core catalog data.
    """
    return test_ctx.require(CORE_PACK)


@pytest.fixture
def graph_ctx(test_ctx: TestContext) -> TestContext:
    """Provide TestContext with CORE_PACK and GRAPH_PACK applied.

    Seeds call graph, import graph, CFG, and DFG tables.

    Returns
    -------
    TestContext
        Context with graph data.
    """
    return test_ctx.require(CORE_PACK, GRAPH_PACK)


@pytest.fixture
def coverage_ctx(test_ctx: TestContext) -> TestContext:
    """Provide TestContext with CORE_PACK and COVERAGE_PACK applied.

    Seeds test catalog, coverage edges, and coverage functions.

    Returns
    -------
    TestContext
        Context with coverage data.
    """
    return test_ctx.require(CORE_PACK, COVERAGE_PACK)


@pytest.fixture
def metrics_ctx(test_ctx: TestContext) -> TestContext:
    """Provide TestContext with CORE_PACK and METRICS_PACK applied.

    Seeds function metrics, risk factors, typedness, and static diagnostics.

    Returns
    -------
    TestContext
        Context with metrics data.
    """
    return test_ctx.require(CORE_PACK, METRICS_PACK)


@pytest.fixture
def full_ctx(test_ctx: TestContext) -> TestContext:
    """Provide TestContext with all seed packs applied.

    Seeds core, graph, coverage, and metrics data for comprehensive tests.

    Returns
    -------
    TestContext
        Context with all data types.
    """
    return test_ctx.require(CORE_PACK, GRAPH_PACK, COVERAGE_PACK, METRICS_PACK)


@pytest.fixture
def scenario_builder() -> type[TestScenario]:
    """Provide the TestScenario builder class for custom scenarios.

    Use this when the standard fixtures don't fit and you need
    custom scenario configuration.

    Returns
    -------
    type[TestScenario]
        The TestScenario class for building custom scenarios.
    """
    return TestScenario


# =============================================================================
# Ingestion Test Fixtures
# =============================================================================
# NOTE: Legacy IngestTestSetup and IngestExecutionContext fixtures removed
# as part of migration to build system. Use TargetExecutionContext for new tests.
