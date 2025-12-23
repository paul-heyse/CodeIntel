"""Pytest configuration for the CodeIntel test suite.

This file defines the *global* fixtures that are reused across many sub-suites.
Most test packages (analytics/serving/storage/...) intentionally reference these
fixtures to avoid duplicated setup code and to keep gateway provisioning
production-parity.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.core.plugins.execution.profiles import DEFAULT_PROFILE_NAME
from tests._helpers import GatewayOptions, ProvisioningConfig, provisioned_gateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.context import create_test_context
from tests._helpers.env import create_provisioned_test_env
from tests._helpers.gateway import GatewayFactory
from tests._helpers.harnesses.analytics_harness import AnalyticsTargetHarness
from tests._helpers.harnesses.graph_harness import GraphTargetHarness
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
from tests._helpers.harnesses.serving_harness import ServingTargetHarness
from tests._helpers.orchestration.coverage_orchestration import (
    create_coverage_edge_env,
    generate_coverage_artifact,
)
from tests._helpers.orchestration.graph_orchestration import (
    create_span_test_env,
    generate_span_coverage,
)
from tests._helpers.orchestration.provisioning import (
    provision_docs_export_ready,
    provision_graph_ready_repo,
)
from tests._helpers.schemas import ensure_storage_contract_catalog
from tests._helpers.seeds import CORE_PACK, COVERAGE_PACK, GRAPH_PACK, METRICS_PACK
from tests._helpers.seeds.architecture import open_seeded_architecture_gateway
from tests._helpers.tool_sandbox import ToolSandbox

ensure_storage_contract_catalog()

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.build.hamilton.runtime import HamiltonRuntime
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.configs import CoverageEdgeEnv, ProvisionedGateway, SpanTestEnv
    from tests._helpers.context import TestContext


@pytest.fixture
def codeintel_env() -> Iterator[None]:
    """Save and restore CODEINTEL_* environment variables."""
    prefix = "CODEINTEL_"
    saved = {key: value for key, value in os.environ.items() if key.startswith(prefix)}
    try:
        yield None
    finally:
        for key in [name for name in os.environ if name.startswith(prefix)]:
            os.environ.pop(key, None)
        os.environ.update(saved)


@pytest.fixture
def fresh_gateway() -> Iterator[StorageGateway]:
    """Provide a fresh in-memory gateway with schema + docs views.

    Yields
    ------
    StorageGateway
        Gateway configured with schemas and docs views; closed after the test.
    """
    gateway = GatewayFactory().open()
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def test_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide an unseeded TestContext with a schema-ready gateway.

    Yields
    ------
    TestContext
        Context with an initialized gateway and filesystem roots.
    """
    ctx = create_test_context(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def core_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a TestContext with core seed pack applied.

    Yields
    ------
    TestContext
        Context seeded with the core pack and closed after the test.
    """
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def graph_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a TestContext with graph seed pack applied.

    Yields
    ------
    TestContext
        Context seeded with the core and graph packs, then closed after the test.
    """
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK, GRAPH_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def coverage_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a TestContext with coverage seed pack applied.

    Yields
    ------
    TestContext
        Context seeded with the core and coverage packs, then closed after the test.
    """
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK, COVERAGE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def metrics_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a TestContext with metrics seed pack applied.

    Yields
    ------
    TestContext
        Context seeded with the core and metrics packs, then closed after the test.
    """
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK, METRICS_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture(scope="session")
def hamilton_runtime() -> HamiltonRuntime:
    """Provide a session-scoped Hamilton runtime for DAG execution.

    Returns
    -------
    HamiltonRuntime
        Runtime instance for DAG execution.
    """
    return build_driver(config={"profile": DEFAULT_PROFILE_NAME})


@pytest.fixture
def build_harness(tmp_path: Path) -> Iterator[HamiltonBuildHarness]:
    """Provide a production-parity Hamilton build harness.

    Yields
    ------
    HamiltonBuildHarness
        Harness instance that is closed after the test.
    """
    with HamiltonBuildHarness.open(tmp_path) as harness:
        yield harness


@pytest.fixture
def graph_target_harness(tmp_path: Path) -> Iterator[GraphTargetHarness]:
    """Provide a graph target harness with a sample repo.

    Yields
    ------
    GraphTargetHarness
        Harness wrapper for graph targets.
    """
    with GraphTargetHarness.open(tmp_path) as harness:
        yield harness


@pytest.fixture
def analytics_target_harness(tmp_path: Path) -> Iterator[AnalyticsTargetHarness]:
    """Provide an analytics target harness with a sample repo.

    Yields
    ------
    AnalyticsTargetHarness
        Harness wrapper for analytics targets.
    """
    with AnalyticsTargetHarness.open(tmp_path) as harness:
        yield harness


@pytest.fixture
def serving_target_harness(tmp_path: Path) -> Iterator[ServingTargetHarness]:
    """Provide a serving target harness with a file-backed gateway.

    Yields
    ------
    ServingTargetHarness
        Harness wrapper for serving targets.
    """
    with ServingTargetHarness.open(tmp_path) as harness:
        yield harness


@pytest.fixture
def tool_sandbox(tmp_path: Path) -> ToolSandbox:
    """Provide a tool sandbox with a stubbed bin directory.

    Returns
    -------
    ToolSandbox
        Sandbox instance with an isolated bin directory.
    """
    return ToolSandbox.create(tmp_path)


@pytest.fixture
def coverage_env(tmp_path: Path) -> Iterator[CoverageEdgeEnv]:
    """Provide a coverage edge environment (repo + gateway + seeded rows).

    Yields
    ------
    CoverageEdgeEnv
        Environment with seeded coverage data; gateway closed after the test.
    """
    env = create_coverage_edge_env(tmp_path)
    try:
        yield env
    finally:
        env.gateway.close()


@pytest.fixture
def coverage_artifact(coverage_env: CoverageEdgeEnv) -> Path:
    """Provide a coverage database artifact for coverage edge tests.

    Returns
    -------
    Path
        Filesystem path to the generated coverage artifact.
    """
    return generate_coverage_artifact(coverage_env).coverage_file


@pytest.fixture
def span_env(tmp_path: Path) -> Iterator[SpanTestEnv]:
    """Provide a span-alignment test environment backed by a real gateway.

    Yields
    ------
    SpanTestEnv
        Span test context built from a snapshot-backed gateway; gateway closed after the test.
    """
    gateway = GatewayFactory().with_snapshot(DEFAULT_REPO, DEFAULT_COMMIT).open()
    try:
        yield create_span_test_env(tmp_path, gateway)
    finally:
        gateway.close()


@pytest.fixture
def span_coverage_artifact(span_env: SpanTestEnv) -> Path:
    """Provide a coverage database artifact for the span-alignment test.

    Returns
    -------
    Path
        Filesystem path to the generated span coverage artifact.
    """
    return generate_span_coverage(span_env.repo_root).coverage_file


@pytest.fixture(scope="session")
def provisioned_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestContext]:
    """Provision a seeded gateway snapshot reused across serving/MCP tests.

    Yields
    ------
    TestContext
        Provisioned context reused across tests; closed after the session.
    """
    repo_root = tmp_path_factory.mktemp("provisioned-repo")
    ctx = create_provisioned_test_env(repo_root)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture(scope="session")
def architecture_gateway() -> Iterator[StorageGateway]:
    """Provision an architecture-focused gateway (docs views + realistic seeds).

    Yields
    ------
    StorageGateway
        Gateway configured for architecture tests; closed after the session.
    """
    provisioned = open_seeded_architecture_gateway(repo=DEFAULT_REPO, commit=DEFAULT_COMMIT)
    try:
        yield provisioned
    finally:
        provisioned.close()


@pytest.fixture
def docs_export_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway seeded for docs export tests.

    Yields
    ------
    ProvisionedGateway
        Gateway prepared for docs export flows; closed after the test.
    """
    ctx = provision_docs_export_ready(tmp_path, file_backed=False)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def graph_ready_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway seeded for graph integration tests.

    Yields
    ------
    ProvisionedGateway
        Gateway prepared for graph tests; closed after the test.
    """
    ctx = provision_graph_ready_repo(tmp_path / "repo", repo=DEFAULT_REPO, commit=DEFAULT_COMMIT)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def loose_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a non-strict gateway for schema drift tests.

    Yields
    ------
    ProvisionedGateway
        Gateway configured with relaxed schema validation; closed after the test.
    """
    config = ProvisioningConfig(
        gateway_options=GatewayOptions(
            apply_schema=True,
            ensure_views=True,
            validate_schema=False,
            strict_schema=False,
            file_backed=False,
        ),
        run_ingestion=False,
    )
    with provisioned_gateway(tmp_path / "repo", config=config) as ctx:
        yield ctx


@pytest.fixture
def coverage_profiles_conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide an isolated in-memory DuckDB connection for profile unit tests.

    Yields
    ------
    duckdb.DuckDBPyConnection
        In-memory DuckDB connection closed after the test.
    """
    con = duckdb.connect(":memory:")
    try:
        yield con
    finally:
        con.close()
