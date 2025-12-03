"""Pytest configuration for the CodeIntel test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.graphs.plugins.builders import callgraph as callgraph_builders
from codeintel.storage.gateway import StorageGateway
from tests._helpers.architecture import open_seeded_architecture_gateway
from tests._helpers.coverage_env import (
    CoverageEdgeEnv,
    create_coverage_edge_env,
    generate_coverage_artifact,
)
from tests._helpers.fixtures import (
    GatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    provision_docs_export_ready,
    provision_graph_ready_repo,
    provision_ingested_repo,
    provisioned_gateway,
)
from tests._helpers.graph_env import SpanTestEnv, create_span_test_env
from tests._helpers.pipeline_env import PipelineEnv, create_pipeline_env

# Compatibility shim for legacy graph step imports that still expect get_callgraph_plugin.
if not hasattr(callgraph_builders, "get_callgraph_plugin"):
    callgraph_builders.get_callgraph_plugin = callgraph_builders.get_callgraph_builder_plugin


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
def pipeline_env(tmp_path: Path) -> Iterator[PipelineEnv]:
    """
    Provide a reusable pipeline environment with seeded catalog data.

    Yields
    ------
    PipelineEnv
        Pipeline environment prepared for graph and coverage integration tests.
    """
    env = create_pipeline_env(tmp_path)
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
def coverage_artifact(coverage_env: CoverageEdgeEnv, tmp_path: Path) -> Iterator[Path]:
    """
    Generate a coverage artifact for the seeded coverage environment.

    Yields
    ------
    Path
        Path to the generated coverage data file.
    """
    artifact = generate_coverage_artifact(coverage_env, coverage_file=tmp_path / ".coverage")
    yield artifact.coverage_file
