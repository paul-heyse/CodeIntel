"""Pytest configuration for the CodeIntel test suite.

This file defines the *global* fixtures that are reused across many sub-suites.
Most test packages (analytics/serving/storage/...) intentionally reference these
fixtures to avoid duplicated setup code and to keep gateway provisioning
production-parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest

from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.gateway import GatewayFactory
from tests._helpers.orchestration.coverage_orchestration import (
    create_coverage_edge_env,
    generate_coverage_artifact,
)
from tests._helpers.orchestration.graph_orchestration import (
    create_span_test_env,
    generate_span_coverage,
)
from tests._helpers.orchestration.provisioning import (
    docs_views_ready_gateway,
    provision_docs_export_ready,
    provision_graph_ready_repo,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.configs import CoverageEdgeEnv, ProvisionedGateway, SpanTestEnv


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
    """Provide a TestContext with core seed pack applied."""
    from tests._helpers.seeds import CORE_PACK

    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def graph_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a TestContext with graph seed pack applied."""
    from tests._helpers.seeds import CORE_PACK, GRAPH_PACK

    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK, GRAPH_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def coverage_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a TestContext with coverage seed pack applied."""
    from tests._helpers.seeds import CORE_PACK, COVERAGE_PACK

    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK, COVERAGE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def metrics_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a TestContext with metrics seed pack applied."""
    from tests._helpers.seeds import CORE_PACK, METRICS_PACK

    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK, METRICS_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def coverage_env(tmp_path: Path) -> Iterator[CoverageEdgeEnv]:
    """Provide a coverage edge environment (repo + gateway + seeded rows)."""
    env = create_coverage_edge_env(tmp_path)
    try:
        yield env
    finally:
        env.gateway.close()


@pytest.fixture
def coverage_artifact(coverage_env: CoverageEdgeEnv) -> Path:
    """Provide a coverage database artifact for coverage edge tests."""
    return generate_coverage_artifact(coverage_env).coverage_file


@pytest.fixture
def span_env(tmp_path: Path) -> Iterator[SpanTestEnv]:
    """Provide a span-alignment test environment backed by a real gateway."""
    gateway = GatewayFactory().with_snapshot(DEFAULT_REPO, DEFAULT_COMMIT).open()
    try:
        yield create_span_test_env(tmp_path, gateway)
    finally:
        gateway.close()


@pytest.fixture
def span_coverage_artifact(span_env: SpanTestEnv) -> Path:
    """Provide a coverage database artifact for the span-alignment test."""
    return generate_span_coverage(span_env.repo_root).coverage_file


@pytest.fixture(scope="session")
def provisioned_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestContext]:
    """Provision a seeded gateway snapshot reused across serving/MCP tests."""
    repo_root = tmp_path_factory.mktemp("provisioned-repo")
    from tests._helpers.env import create_provisioned_test_env

    ctx = create_provisioned_test_env(repo_root)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture(scope="session")
def architecture_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[StorageGateway]:
    """Provision an architecture-focused gateway (docs views + realistic seeds)."""
    repo_root = tmp_path_factory.mktemp("architecture")
    provisioned = docs_views_ready_gateway(repo_root, repo=DEFAULT_REPO, commit=DEFAULT_COMMIT)
    try:
        yield provisioned.gateway
    finally:
        provisioned.close()


@pytest.fixture
def docs_export_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway seeded for docs export tests."""
    ctx = provision_docs_export_ready(tmp_path, file_backed=False)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def graph_ready_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway seeded for graph integration tests."""
    ctx = provision_graph_ready_repo(tmp_path / "repo", repo=DEFAULT_REPO, commit=DEFAULT_COMMIT)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def loose_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a non-strict gateway for schema drift tests."""
    from tests._helpers import GatewayOptions, ProvisioningConfig, provisioned_gateway

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
    """Provide an isolated in-memory DuckDB connection for profile unit tests."""
    con = duckdb.connect(":memory:")
    try:
        yield con
    finally:
        con.close()
