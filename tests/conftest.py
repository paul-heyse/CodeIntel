"""Pytest configuration for the CodeIntel test suite.

This file defines the *global* fixtures that are reused across many sub-suites.
Most test packages (analytics/serving/storage/...) intentionally reference these
fixtures to avoid duplicated setup code and to keep gateway provisioning
production-parity.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.providers import create_default_providers
from codeintel.build.schemas import configure_contract_service, configure_schema_service
from codeintel.build.settings import DEFAULT_PROFILE_NAME
from codeintel.config.models import ToolsConfig
from codeintel.observability.runtime import shutdown_observability
from codeintel.runtime.compose import compose_runtime
from codeintel.runtime.runtime_bundle import RuntimeBundle
from tests._helpers import (
    GatewayOptions,
    ProvisioningConfig,
    TestScenario,
    provisioned_gateway,
)
from tests._helpers.build import TEST_BUILD_SETTINGS
from tests._helpers.columnar_streams import (
    contract_schema_for_table_key as contract_schema_for_table_key_fn,
)
from tests._helpers.columnar_streams import (
    reader_for_rows as reader_for_rows_fn,
)
from tests._helpers.env import create_provisioned_test_env
from tests._helpers.fixtures.rows import columnar_rows_for as columnar_rows_for_fn
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT
from tests._helpers.gateway import GatewayFactory
from tests._helpers.harnesses.analytics_harness import AnalyticsTargetHarness
from tests._helpers.harnesses.graph_harness import GraphTargetHarness
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness, HarnessConfig
from tests._helpers.harnesses.serving_harness import ServingTargetHarness
from tests._helpers.orchestration.graph_orchestration import (
    create_span_test_env,
)
from tests._helpers.orchestration.provisioning import (
    provision_docs_export_ready,
    provision_graph_ready_repo,
)
from tests._helpers.pytest_options import apply_pytest_options, register_pytest_options
from tests._helpers.schemas import ensure_schema_service, ensure_storage_contract_catalog
from tests._helpers.seeds.architecture import open_seeded_architecture_gateway
from tests._helpers.serving_snapshot_factory import ServingSnapshot, ServingSnapshotFactory
from tests._helpers.tooling_audit import ToolCallLog
from tests._helpers.tooling_audit import require_tooling as _require_tooling
from tests._helpers.waiting import (
    eventually as eventually_fn,
)
from tests._helpers.waiting import (
    eventually_async as eventually_async_fn,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path

    import pyarrow as pa

    from codeintel.core.columnar.rows import ColumnarRows
    from codeintel.core.schemas import SchemaService
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.configs import ProvisionedGateway, SpanTestEnv
    from tests._helpers.context import TestContext


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for the pytest session."""
    register_pytest_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Apply configured pytest options and register markers at session startup."""
    apply_pytest_options(config)
    config.addinivalue_line(
        "markers",
        "requires_tools(*names): mark tests that need external tool binaries",
    )


def pytest_sessionfinish(
    session: pytest.Session,
    exitstatus: int,
) -> None:
    """Shutdown observability once per test session."""
    _ = session
    _ = exitstatus
    shutdown_observability()


@pytest.fixture
def parity_harness_config() -> HarnessConfig:
    """Provide production-parity HarnessConfig defaults.

    Returns
    -------
    HarnessConfig
        Default harness configuration for parity tests.
    """
    return HarnessConfig(repo=DEFAULT_VARIANT.repo, commit=DEFAULT_VARIANT.commit)


@pytest.fixture
def serving_snapshot_factory(tmp_path: Path) -> ServingSnapshotFactory:
    """Provide a ServingSnapshotFactory bound to the test temp directory.

    Returns
    -------
    ServingSnapshotFactory
        Factory rooted at the temporary directory.
    """
    return ServingSnapshotFactory(tmp_path)


@pytest.fixture
def serving_snapshot(serving_snapshot_factory: ServingSnapshotFactory) -> ServingSnapshot:
    """Provide a demo serving snapshot on disk.

    Returns
    -------
    ServingSnapshot
        Snapshot pointing to demo data on disk.
    """
    return serving_snapshot_factory.demo_snapshot()


@pytest.fixture
def eventually() -> Callable[..., object]:
    """Provide the eventually helper as a fixture.

    Returns
    -------
    Callable[..., object]
        Synchronous eventually helper.
    """
    return eventually_fn


@pytest.fixture
def eventually_async() -> Callable[..., object]:
    """Provide the async eventually helper as a fixture.

    Returns
    -------
    Callable[..., object]
        Asynchronous eventually helper.
    """
    return eventually_async_fn


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
    ctx = TestScenario().build(tmp_path)
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
    ctx = TestScenario.minimal().build(tmp_path)
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
    ctx = TestScenario.with_graph().build(tmp_path)
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
    ctx = TestScenario.with_metrics().build(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture(scope="session")
def runtime_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BuildEnv]:
    """Provide a session-scoped BuildEnv for runtime composition.

    Yields
    ------
    BuildEnv
        Build environment for runtime composition.
    """
    ctx = TestScenario.minimal().build(tmp_path_factory.mktemp("hamilton-runtime"))
    providers = create_default_providers(ToolsConfig.default())
    env = BuildEnv(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
        providers=providers,
        config=BuildConfig.empty(),
        settings=TEST_BUILD_SETTINGS,
        profile=DEFAULT_PROFILE_NAME,
    )
    try:
        yield env
    finally:
        ctx.close()


@pytest.fixture(scope="session")
def hamilton_runtime(runtime_env: BuildEnv) -> RuntimeBundle:
    """Provide a session-scoped runtime bundle for DAG inspection.

    Returns
    -------
    RuntimeBundle
        Runtime bundle with driver, catalog, and tag query.
    """
    config: dict[str, object] = {}
    if runtime_env.profile:
        config["profile"] = runtime_env.profile
    config.update(runtime_env.variants.as_hamilton_config())
    config["variant_fingerprint"] = runtime_env.variants.variant_fingerprint
    return compose_runtime(env=runtime_env, config=config).bundle


def _should_skip_session_services(request: pytest.FixtureRequest) -> bool:
    items = request.session.items
    if not items:
        return False
    return all(item.get_closest_marker("no_runtime_env") is not None for item in items)


@pytest.fixture(scope="session", autouse=True)
def _session_schema_service(request: pytest.FixtureRequest) -> None:
    if _should_skip_session_services(request):
        return

    runtime: RuntimeBundle = request.getfixturevalue("hamilton_runtime")
    configure_schema_service(runtime=runtime)
    configure_contract_service(runtime=runtime)
    ensure_storage_contract_catalog()


@pytest.fixture(scope="session")
def schema_service() -> SchemaService:
    """Provide the configured SchemaService for contract-aware helpers.

    Returns
    -------
    SchemaService
        Initialized schema service instance.
    """
    return ensure_schema_service()


@pytest.fixture(scope="session")
def contract_schema_for(
    schema_service: SchemaService,
) -> Callable[[str], pa.Schema]:
    """Provide a contract schema resolver for table keys.

    Returns
    -------
    Callable[[str], pyarrow.Schema]
        Resolver that returns an Arrow contract schema.
    """
    _ = schema_service
    return contract_schema_for_table_key_fn


@pytest.fixture
def columnar_rows_for(
    schema_service: SchemaService,
) -> Callable[[str, Sequence[Mapping[str, object]]], ColumnarRows]:
    """Provide a columnar row factory aligned to schema contracts.

    Returns
    -------
    Callable[[str, Sequence[Mapping[str, object]]], ColumnarRows]
        Columnar row factory for table keys.
    """
    _ = schema_service
    return columnar_rows_for_fn


@pytest.fixture
def reader_for_rows(
    schema_service: SchemaService,
) -> Callable[..., pa.RecordBatchReader]:
    """Provide a RecordBatchReader factory aligned to schema contracts.

    Returns
    -------
    Callable[..., pyarrow.RecordBatchReader]
        Reader factory for table keys.
    """
    _ = schema_service
    return reader_for_rows_fn


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
def tool_call_log(tmp_path: Path) -> Iterator[ToolCallLog]:
    """Provide a per-test tool invocation log file.

    Yields
    ------
    ToolCallLog
        Log wrapper for recorded tool calls.
    """
    path = tmp_path / "tool_calls.jsonl"
    env_key = "CODEINTEL_TOOL_CALL_LOG"
    previous = os.environ.get(env_key)
    os.environ[env_key] = str(path)
    try:
        yield ToolCallLog(path)
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


@pytest.fixture(scope="session")
def require_tooling() -> None:
    """Explicitly verify tool binaries for tests that opt in."""
    _require_tooling()


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip or fail tests when required tool binaries are missing."""
    marker = item.get_closest_marker("requires_tools")
    if marker is None:
        return
    tools = tuple(str(arg) for arg in marker.args)
    if not tools:
        return
    _require_tooling(required_tools=tools)


@pytest.fixture
def span_env(tmp_path: Path) -> Iterator[SpanTestEnv]:
    """Provide a span-alignment test environment backed by a real gateway.

    Yields
    ------
    SpanTestEnv
        Span test context built from a snapshot-backed gateway; gateway closed after the test.
    """
    gateway = GatewayFactory().with_snapshot(DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit).open()
    try:
        yield create_span_test_env(tmp_path, gateway)
    finally:
        gateway.close()


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
    provisioned = open_seeded_architecture_gateway(
        repo=DEFAULT_VARIANT.repo, commit=DEFAULT_VARIANT.commit
    )
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
    ctx = provision_graph_ready_repo(
        tmp_path / "repo", repo=DEFAULT_VARIANT.repo, commit=DEFAULT_VARIANT.commit
    )
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
