"""Tests for DocstringsIngestPlugin and basic fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute import DocstringsExtractStep
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.docstrings_plugin import DocstringsIngestPlugin
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort
from codeintel.ingestion.ports.storage import IngestStoragePort
from codeintel.storage.gateway import StorageGateway
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.env import create_test_env
from tests._helpers.env_options import EnvOptions
from tests._helpers.fakes.contexts import (
    EnvOverrides,
    ExecutionContextBuilder,
    TargetResourceOverrides,
    make_test_output_target,
)
from tests._helpers.fakes.ingestion_plugins import (
    RecordingDiscoveryAdapter,
    RecordingStep,
    RecordingStorageAdapter,
    StepCallCapture,
)
from tests._helpers.fakes.recording_gateways import (
    ConnectionRecordingGateway,
    FailingGateway,
)


def _make_plugin(
    capture: StepCallCapture,
    *,
    table_key: str = "core.docstrings",
    result: StepResult | None = None,
) -> DocstringsIngestPlugin:
    """Create a DocstringsIngestPlugin with recording adapters.

    Parameters
    ----------
    capture
        Capture object to record adapter and step calls.
    table_key
        Table key to use in the step.
    result
        Optional custom result to return from the step.

    Returns
    -------
    DocstringsIngestPlugin
        Configured plugin instance.
    """
    return DocstringsIngestPlugin(
        storage_adapter_factory=lambda gateway: RecordingStorageAdapter(gateway, capture=capture),
        discovery_adapter_factory=lambda repo_root: RecordingDiscoveryAdapter(
            repo_root, capture=capture
        ),
        step_factory=_build_step_factory(
            capture=capture,
            table_key=table_key,
            result=result,
        ),
    )


def _build_step_factory(
    *,
    capture: StepCallCapture,
    table_key: str,
    result: StepResult | None,
) -> Callable[
    [
        IngestStoragePort,
        ModuleDiscoveryPort,
    ],
    DocstringsExtractStep,
]:
    """Build a step factory that creates recording steps.

    Parameters
    ----------
    capture
        Capture object to record step calls.
    table_key
        Table key for the step.
    result
        Optional custom result.

    Returns
    -------
    Callable
        Factory function for creating recording steps.
    """

    def _factory(
        storage: IngestStoragePort, discovery: ModuleDiscoveryPort
    ) -> DocstringsExtractStep:
        return cast(
            "DocstringsExtractStep",
            RecordingStep(
                storage,
                discovery,
                capture=capture,
                table_key=table_key,
                result=result,
            ),
        )

    return _factory


def _build_target_context(
    tmp_path: Path,
    plugin: DocstringsIngestPlugin,
    *,
    repo_root: Path,
    modules: tuple[str, ...] = (),
    gateway: StorageGateway | ConnectionRecordingGateway | FailingGateway | None = None,
) -> TargetExecutionContext:
    """Build a TargetExecutionContext for ingestion plugin testing.

    Parameters
    ----------
    tmp_path
        Temporary directory for test isolation.
    plugin
        Plugin to build context for.
    repo_root
        Repository root directory.
    modules
        Module paths to include in resources.
    gateway
        Optional custom gateway (recording or failing).

    Returns
    -------
    TargetExecutionContext
        Configured context for plugin execution.
    """
    if gateway is None:
        env = create_test_env(tmp_path, options=EnvOptions(repo_root=repo_root))
        gateway = env.gateway

    # Use repo_root for the snapshot, not tmp_path
    builder = ExecutionContextBuilder.create(
        repo_root,
        env_overrides=EnvOverrides(
            gateway=cast("StorageGateway", gateway),
            snapshot=(DEFAULT_REPO, DEFAULT_COMMIT),
        ),
    )
    target = make_test_output_target(plugin)
    return builder.build_target_context(
        target,
        resources=TargetResourceOverrides(modules=modules),
    )


@pytest.mark.anyio
async def test_execute_invokes_step_and_returns_row_counts(tmp_path: Path) -> None:
    """Happy path: modules from resources flow through adapters to the step."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    captured = StepCallCapture()
    plugin = _make_plugin(captured)

    ctx = _build_target_context(
        tmp_path,
        plugin,
        repo_root=repo_root,
        modules=("pkg/mod.py",),
    )
    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts, {"core.docstrings": 1})
    expect_true(captured.storage is not None)
    expect_true(isinstance(captured.storage, RecordingStorageAdapter))
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/mod.py")
    expect_equal(module_record.file_path, repo_root / "pkg/mod.py")


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(tmp_path: Path) -> None:
    """When modules are absent in resources, the gateway should be queried."""
    repo_root = build_repo_tree(tmp_path / "repo", {})

    # Create a real gateway and seed test data
    env = create_test_env(tmp_path / "env", options=EnvOptions(repo_root=repo_root))
    env.gateway.con.execute(
        "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
        ["pkg.db_mod", "pkg/db_mod.py", DEFAULT_REPO, DEFAULT_COMMIT],
    )

    # Wrap with ConnectionRecordingGateway to track con.execute() calls
    recording_gateway = ConnectionRecordingGateway(env.gateway)
    captured = StepCallCapture()
    plugin = _make_plugin(captured)

    try:
        ctx = _build_target_context(
            tmp_path,
            plugin,
            repo_root=repo_root,
            modules=(),
            gateway=recording_gateway,
        )
        result = await plugin.execute(ctx)

        expect_equal(result.row_counts, {"core.docstrings": 1})
        # Verify the SQL query was made
        sql, params = recording_gateway.executions[0]
        expect_true("core.modules" in sql)
        expect_equal(params, [DEFAULT_REPO, DEFAULT_COMMIT])
        module_record = captured.modules[0]
        expect_equal(module_record.rel_path, "pkg/db_mod.py")
        expect_equal(module_record.file_path, repo_root / "pkg/db_mod.py")
        expect_equal(captured.repo, DEFAULT_REPO)
        expect_equal(captured.commit, DEFAULT_COMMIT)
    finally:
        env.close()


@pytest.mark.anyio
async def test_execute_recovers_from_gateway_errors(tmp_path: Path) -> None:
    """Database lookup failures should result in an empty module set."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    # Use a FailingGateway that raises RuntimeError on execute
    gateway = FailingGateway(error_message="db down")
    captured = StepCallCapture()
    plugin = _make_plugin(captured)

    ctx = _build_target_context(
        tmp_path,
        plugin,
        repo_root=repo_root,
        modules=(),
        gateway=gateway,
    )
    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(captured.modules, [])
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
