"""Shared wiring scenario helpers for ingestion plugins."""

from __future__ import annotations

from collections.abc import Callable
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
    scenario: str,
    plugin_factory: Callable[[StepCallCapture], TargetPlugin],
    tmp_path: Path,
    *,
    table_key: str,
) -> None:
    """Execute a standard wiring scenario for a synchronous ingest plugin.

    Raises
    ------
    ValueError
        If an unknown scenario is provided.
    """
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/mod.py": "x = 1\n"},
    )

    if scenario == "resources":
        capture = StepCallCapture()
        plugin = plugin_factory(capture)
        ctx = build_target_context_for_plugin(
            plugin,
            tmp_path,
            config=TargetContextConfig(
                repo_root=repo_root,
                resources=TargetResourceOverrides(modules=("pkg/mod.py",)),
            ),
        )

        result = await plugin.execute(ctx)

        expect_true(result.success)
        expect_equal(result.row_counts, {table_key: 1})
        expect_equal([record.rel_path for record in capture.modules], ["pkg/mod.py"])
        expect_equal(capture.repo, ctx.repo)
        expect_equal(capture.commit, ctx.commit)
        expect_equal(capture.repo_root, repo_root)
        return

    if scenario == "db_fallback":
        capture = StepCallCapture()
        plugin = plugin_factory(capture)
        gateway = GatewayFactory().with_macros().open()
        recording_gateway = ConnectionRecordingGateway(gateway)
        try:
            ctx = build_target_context_for_plugin(
                plugin,
                tmp_path,
                config=TargetContextConfig(
                    repo_root=repo_root,
                    gateway=recording_gateway,
                    resources=TargetResourceOverrides(modules=()),
                ),
            )
            seed_modules_and_repo_map(ctx, ["pkg/db_mod.py"])

            result = await plugin.execute(ctx)

            expect_true(result.success)
            expect_equal(result.row_counts, {table_key: 1})
            expect_true(recording_gateway.executions)
            expect_equal([record.rel_path for record in capture.modules], ["pkg/db_mod.py"])
            expect_equal(capture.repo, ctx.repo)
            expect_equal(capture.commit, ctx.commit)
            expect_equal(capture.repo_root, repo_root)
        finally:
            recording_gateway.close()
        return

    if scenario == "gateway_failure":
        capture = StepCallCapture()
        plugin = plugin_factory(capture)
        failing_gateway = FailingGateway(GatewayFactory().with_macros().open(), "db down")
        try:
            ctx = build_target_context_for_plugin(
                plugin,
                tmp_path,
                config=TargetContextConfig(
                    repo_root=repo_root,
                    gateway=cast("StorageGateway", failing_gateway),
                    resources=TargetResourceOverrides(modules=()),
                ),
            )

            result = await plugin.execute(ctx)

            expect_true(result.success)
            expect_equal(result.row_counts.get(table_key, 0), 0)
            expect_equal(capture.modules, [])
            expect_equal(capture.repo, ctx.repo)
            expect_equal(capture.commit, ctx.commit)
            expect_equal(capture.repo_root, repo_root)
            return
        finally:
            failing_gateway.close()

    message = f"Unknown scenario '{scenario}'"
    raise ValueError(message)


def run_module_path_resolution_scenarios(
    plugin_factory: Callable[[StepCallCapture], TargetPlugin],
    get_module_paths: Callable[[TargetExecutionContext], list[str]],
    tmp_path: Path,
    *,
    resources_path: str,
    scenario: str | None = None,
) -> None:
    """Exercise module path resolution across resources, DB, and gateway failure.

    Raises
    ------
    ValueError
        If an unknown scenario is provided.
    """
    capture = StepCallCapture()
    plugin = plugin_factory(capture)

    scenarios = {
        "resources": lambda: _assert_resources_path(
            plugin, get_module_paths, tmp_path, resources_path
        ),
        "db_fallback": lambda: _assert_db_path(plugin, get_module_paths, tmp_path, resources_path),
        "gateway_failure": lambda: _assert_gateway_failure(plugin, get_module_paths, tmp_path),
    }
    if scenario is not None:
        run = scenarios.get(scenario)
        if run is None:
            message = f"Unknown scenario '{scenario}'"
            raise ValueError(message)
        run()
        return

    for run in scenarios.values():
        run()


def _assert_resources_path(
    plugin: TargetPlugin,
    get_module_paths: Callable[[TargetExecutionContext], list[str]],
    tmp_path: Path,
    resources_path: str,
) -> None:
    overrides = TargetResourceOverrides(modules=(resources_path,))
    ctx = build_target_context_for_plugin(
        plugin, tmp_path, config=TargetContextConfig(resources=overrides)
    )
    expect_equal(get_module_paths(ctx), [resources_path])


def _assert_db_path(
    plugin: TargetPlugin,
    get_module_paths: Callable[[TargetExecutionContext], list[str]],
    tmp_path: Path,
    resources_path: str,
) -> None:
    ctx_db = build_target_context_for_plugin(plugin, tmp_path)
    records = module_records_for_paths([resources_path], ctx_db.repo_root)
    seed_modules_and_repo_map(ctx_db, [record.rel_path for record in records])
    expect_equal(get_module_paths(ctx_db), [resources_path])


def _assert_gateway_failure(
    plugin: TargetPlugin,
    get_module_paths: Callable[[TargetExecutionContext], list[str]],
    tmp_path: Path,
) -> None:
    failing_gateway = FailingGateway(GatewayFactory().with_macros().open(), "db down")
    try:
        ctx_fail = build_target_context_for_plugin(
            plugin,
            tmp_path,
            config=TargetContextConfig(
                gateway=cast("StorageGateway", failing_gateway),
                resources=TargetResourceOverrides(modules=()),
            ),
        )
        expect_equal(get_module_paths(ctx_fail), [])
    finally:
        failing_gateway.close()
