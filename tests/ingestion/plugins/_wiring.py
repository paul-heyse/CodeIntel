"""Shared wiring scenario helpers for ingestion plugins."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from codeintel.build.plugin import TargetPlugin
from tests._helpers import build_repo_tree
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.ingestion_plugins import StepCallCapture
from tests._helpers.fakes.recording_gateways import ConnectionRecordingGateway, FailingGateway
from tests._helpers.gateway import GatewayFactory
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_target_context_for_plugin,
    seed_modules_and_repo_map,
)


async def run_sync_plugin_wiring_scenario(
    scenario: str,
    plugin_factory: Callable[[StepCallCapture], TargetPlugin],
    tmp_path: Path,
    *,
    table_key: str,
) -> None:
    """Execute a standard wiring scenario for a synchronous ingest plugin."""
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
        failing_gateway = FailingGateway("db down")
        ctx = build_target_context_for_plugin(
            plugin,
            tmp_path,
            config=TargetContextConfig(
                repo_root=repo_root,
                gateway=failing_gateway,
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

    raise ValueError(f"Unknown scenario '{scenario}'")
