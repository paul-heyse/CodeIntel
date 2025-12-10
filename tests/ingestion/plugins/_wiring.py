"""Shared wiring scenario helpers for ingestion plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.plugin import TargetPlugin
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.ingestion_plugins import StepCallCapture
from tests._helpers.fakes.recording_gateways import ConnectionRecordingGateway, FailingGateway
from tests._helpers.gateway import GatewayFactory
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_repo_tree,
    build_target_context_for_plugin,
    module_records_for_paths,
    seed_modules_and_repo_map,
)

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway


async def run_sync_plugin_wiring_scenario(
    plugin_factory: Callable[[StepCallCapture], TargetPlugin],
    tmp_path: Path,
    *,
    table_key: str,
    options: Mapping[str, bool],
) -> None:
    """Execute a standard wiring scenario for a synchronous ingest plugin."""
    await _run_resource_case(
        plugin_factory,
        tmp_path,
        table_key=table_key,
        options=options,
    )


def run_module_path_resolution_scenarios(
    plugin_factory: Callable[[StepCallCapture], TargetPlugin],
    get_module_paths: Callable[[TargetExecutionContext], list[str]],
    tmp_path: Path,
    *,
    resources_path: str,
    options: Mapping[str, bool],
) -> None:
    """Exercise module path resolution across resources, DB, and gateway failure."""
    _run_module_path_case(
        plugin_factory,
        get_module_paths,
        tmp_path,
        resources_path=resources_path,
        options=options,
    )


async def _run_resource_case(
    plugin_factory: Callable[[StepCallCapture], TargetPlugin],
    tmp_path: Path,
    *,
    table_key: str,
    options: Mapping[str, bool],
) -> None:
    simulate_resources = options["simulate_resources"]
    simulate_db_fallback = options["simulate_db_fallback"]
    simulate_gateway_failure = options["simulate_gateway_failure"]
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/mod.py": "x = 1\n"},
    )
    capture = StepCallCapture()
    plugin = plugin_factory(capture)
    base_gateway = GatewayFactory().with_macros().open()
    gateway: StorageGateway
    recording_gateway: ConnectionRecordingGateway | None = None
    if simulate_gateway_failure:
        gateway = cast("StorageGateway", FailingGateway(base_gateway, "db down"))
    elif simulate_db_fallback:
        recording_gateway = ConnectionRecordingGateway(base_gateway)
        gateway = recording_gateway
    else:
        gateway = base_gateway

    resources = TargetResourceOverrides(
        modules=("pkg/mod.py",) if simulate_resources else (),
    )
    try:
        ctx = build_target_context_for_plugin(
            plugin,
            tmp_path,
            config=TargetContextConfig(
                repo_root=repo_root,
                gateway=gateway,
                resources=resources,
            ),
        )
        if simulate_db_fallback:
            seed_modules_and_repo_map(ctx, ["pkg/db_mod.py"])

        result = await plugin.execute(ctx)
    finally:
        gateway.close()

    expected_modules: list[str]
    if simulate_gateway_failure:
        expected_modules = []
    elif simulate_resources:
        expected_modules = ["pkg/mod.py"]
    elif simulate_db_fallback:
        expected_modules = ["pkg/db_mod.py"]
    else:
        expected_modules = []

    expect_true(result.success)
    expect_equal(result.row_counts.get(table_key, 0), len(expected_modules))
    if simulate_db_fallback and recording_gateway is not None:
        expect_true(recording_gateway.executions)
    expect_equal([record.rel_path for record in capture.modules], expected_modules)
    expect_equal(capture.repo, ctx.repo)
    expect_equal(capture.commit, ctx.commit)
    expect_equal(capture.repo_root, repo_root)


def _run_module_path_case(
    plugin_factory: Callable[[StepCallCapture], TargetPlugin],
    get_module_paths: Callable[[TargetExecutionContext], list[str]],
    tmp_path: Path,
    *,
    resources_path: str,
    options: Mapping[str, bool],
) -> None:
    simulate_resources = options["simulate_resources"]
    simulate_db_fallback = options["simulate_db_fallback"]
    simulate_gateway_failure = options["simulate_gateway_failure"]
    capture = StepCallCapture()
    plugin = plugin_factory(capture)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    base_gateway = GatewayFactory().with_macros().open()
    gateway: StorageGateway
    if simulate_gateway_failure:
        gateway = cast("StorageGateway", FailingGateway(base_gateway, "db down"))
    else:
        gateway = base_gateway

    resources = TargetResourceOverrides(
        modules=(resources_path,) if simulate_resources else (),
    )
    try:
        ctx = build_target_context_for_plugin(
            plugin,
            tmp_path,
            config=TargetContextConfig(
                repo_root=repo_root,
                gateway=gateway,
                resources=resources,
            ),
        )
        if simulate_db_fallback:
            records = module_records_for_paths([resources_path], ctx.repo_root)
            seed_modules_and_repo_map(ctx, [record.rel_path for record in records])
        expected_modules = [resources_path] if (simulate_resources or simulate_db_fallback) else []
        expect_equal(get_module_paths(ctx), expected_modules)
    finally:
        gateway.close()
