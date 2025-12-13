"""Native SCIP ingestion with Hamilton subgraph.

This module implements SCIP indexing as a native Hamilton pipeline with:
- tool__scip: Execute scip-python to generate index
- parse__scip: Parse SCIP index into tables
- t__scip: Orchestrate execution and return TargetRunRecord
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.outputs import expected_artifacts
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    create_run_record,
    save_manifest,
)
from codeintel.build.hamilton.native.tools import ToolExecutionResult, ToolExecutionSpec
from codeintel.build.hamilton.native.tools.executor import execute_tool

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph


@tag(domain="ingestion", target="scip", node_kind="tool")
def tool__scip(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> ToolExecutionResult:
    """Execute scip-python tool to generate SCIP index.

    Parameters
    ----------
    env
        Build environment with paths and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    ToolExecutionResult
        Tool execution result with artifact reference.
    """
    if t__modules.status != "succeeded":
        return ToolExecutionResult(
            success=False,
            artifact=None,
            duration_ms=0.0,
            stdout="",
            stderr="Upstream modules target failed",
            return_code=-1,
        )

    output_path = env.paths.scip_dir / "index.scip"

    spec = ToolExecutionSpec(
        tool_name="scip-python",
        command_args=(
            "index",
            "--project-name",
            env.snapshot.repo,
            "--output",
            str(output_path),
            str(env.snapshot.repo_root),
        ),
        output_path=output_path,
        timeout_seconds=600.0,
    )

    return execute_tool(spec, env)


@tag(domain="ingestion", target="scip", node_kind="parse")
def parse__scip(
    env: BuildEnv,
    tool__scip: ToolExecutionResult,
) -> dict[str, object]:
    """Parse SCIP index into structured data.

    Parameters
    ----------
    env
        Build environment.
    tool__scip
        Tool execution result with SCIP index artifact.

    Returns
    -------
    dict[str, object]
        Parsed SCIP data for downstream processing. Contains "success" key
        and either "data" (on success) or "error" (on failure).
    """
    if not tool__scip.success or tool__scip.artifact is None:
        return {"success": False, "error": tool__scip.stderr}

    scip_path = tool__scip.artifact.path
    if scip_path is None:
        return {"success": False, "error": "SCIP artifact path is None"}

    return {
        "success": True,
        "data": {
            "scip_path": scip_path,
            "repo": env.snapshot.repo,
            "commit": env.snapshot.commit,
        },
    }


@tag(domain="ingestion", target="scip", node_kind="target")
def t__scip(
    env: BuildEnv,
    graph: TargetGraph,
    tool__scip: ToolExecutionResult,
    parse__scip: dict[str, object],
    t__modules: TargetRunRecord,
) -> TargetRunRecord:
    """Orchestrate SCIP target execution.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph for metadata.
    tool__scip
        Tool execution result.
    parse__scip
        Parsed SCIP data.
    t__modules
        Upstream modules result.

    Returns
    -------
    TargetRunRecord
        Complete target execution record.
    """
    target = graph.get("scip")

    # Check for upstream failure
    if t__modules.status != "succeeded":
        return TargetRunRecord(
            target="scip",
            plugin_name="native:scip",
            status="skipped",
            input_hash=None,
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error="upstream_failed",
            datasets=(),
            artifacts=expected_artifacts(
                target,
                env.snapshot,
                path_formatter={
                    "build_dir": str(env.paths.build_dir),
                    "scip_dir": str(env.paths.scip_dir),
                    "repo_root": str(env.snapshot.repo_root),
                },
            ) if target else (),
        )

    executor = NativeTargetExecutor.for_target(env, graph, "scip")

    # Check skip logic
    if executor.should_skip():
        return executor.skip()

    # Check tool execution
    if not tool__scip.success:
        return executor.fail(RuntimeError(tool__scip.stderr or "Tool execution failed"))

    # Check parsing
    if not parse__scip.get("success"):
        return executor.fail(RuntimeError(str(parse__scip.get("error", "Parse failed"))))

    # Success case - materialization would happen here in full implementation
    run = NativeRunInfo(
        input_hash=executor.input_hash,
        options_hash=executor.options_hash,
        duration_ms=0.0,
    )
    record = create_run_record(
        executor.target,
        "succeeded",
        executor.input_hash,
        env=env,
        run=run,
    )

    save_manifest(env, record)
    return record


__all__ = [
    "parse__scip",
    "t__scip",
    "tool__scip",
]
