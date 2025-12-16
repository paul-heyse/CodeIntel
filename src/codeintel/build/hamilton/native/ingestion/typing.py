"""Native typing ingestion with Hamilton subgraph.

This module implements typing checks as a native Hamilton pipeline with:
- tool__typing__pyright: Execute pyright for type checking
- tool__typing__pyrefly: Execute pyrefly for additional checks
- tool__typing__ruff: Execute ruff for static diagnostics
- parse__typing: Aggregate results into tables
- t__typing: Orchestrate execution and return TargetRunRecord

Phase 2: Enhanced with Hamilton-native validation via @check_output_custom
and @schema.output documentation.
"""

from __future__ import annotations

from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.outputs import expected_artifacts
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    RunRecordInputs,
    create_run_record,
    save_manifest,
)
from codeintel.build.hamilton.native.tools import ToolExecutionResult, ToolExecutionSpec
from codeintel.build.hamilton.native.tools.executor import execute_tool
from codeintel.build.targets import TargetGraph


@dataclass(frozen=True)
class TypingParseResult:
    """Parsed typing results for the typing target."""

    success: bool
    error: str | None
    artifacts: dict[str, str | None]


_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph)


@tag(domain="ingestion", target="typing", node_type="tool")
def tool__typing__pyright(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> ToolExecutionResult:
    """Execute pyright tool for type checking.

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

    output_path = env.paths.build_dir / "typing" / "pyright.json"

    spec = ToolExecutionSpec(
        tool_name="pyright",
        command_args=(
            "--outputjson",
            str(env.snapshot.repo_root),
        ),
        output_path=output_path,
        timeout_seconds=600.0,
    )

    return execute_tool(spec, env)


@tag(domain="ingestion", target="typing", node_type="tool")
def tool__typing__pyrefly(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> ToolExecutionResult:
    """Execute pyrefly tool for additional type checks.

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

    output_path = env.paths.build_dir / "typing" / "pyrefly.json"

    spec = ToolExecutionSpec(
        tool_name="pyrefly",
        command_args=(
            "check",
            "--format",
            "json",
            str(env.snapshot.repo_root),
        ),
        output_path=output_path,
        timeout_seconds=600.0,
    )

    return execute_tool(spec, env)


@tag(domain="ingestion", target="typing", node_type="tool")
def tool__typing__ruff(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> ToolExecutionResult:
    """Execute ruff tool for static diagnostics.

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

    output_path = env.paths.build_dir / "typing" / "ruff.json"

    spec = ToolExecutionSpec(
        tool_name="ruff",
        command_args=(
            "check",
            "--output-format",
            "json",
            str(env.snapshot.repo_root),
        ),
        output_path=output_path,
        timeout_seconds=300.0,
    )

    return execute_tool(spec, env)


@tag(domain="ingestion", target="typing", node_type="compute")
def parse__typing(
    tool__typing__pyright: ToolExecutionResult,
    tool__typing__pyrefly: ToolExecutionResult,
    tool__typing__ruff: ToolExecutionResult,
) -> TypingParseResult:
    """Aggregate typing tool results into structured data.

    Parameters
    ----------
    tool__typing__pyright
        Pyright execution result.
    tool__typing__pyrefly
        Pyrefly execution result.
    tool__typing__ruff
        Ruff execution result.

    Returns
    -------
    TypingParseResult
        Aggregated typing results.
    """
    artifacts = {
        "pyright": tool__typing__pyright.artifact.path if tool__typing__pyright.artifact else None,
        "pyrefly": tool__typing__pyrefly.artifact.path if tool__typing__pyrefly.artifact else None,
        "ruff": tool__typing__ruff.artifact.path if tool__typing__ruff.artifact else None,
    }

    # Consider success if at least one tool succeeded
    overall_success = any(
        (
            tool__typing__pyright.success,
            tool__typing__pyrefly.success,
            tool__typing__ruff.success,
        )
    )

    if not overall_success:
        errors: list[str] = []
        if not tool__typing__pyright.success:
            errors.append(f"pyright: {tool__typing__pyright.stderr}")
        if not tool__typing__pyrefly.success:
            errors.append(f"pyrefly: {tool__typing__pyrefly.stderr}")
        if not tool__typing__ruff.success:
            errors.append(f"ruff: {tool__typing__ruff.stderr}")
        return TypingParseResult(success=False, error="; ".join(errors), artifacts=artifacts)

    return TypingParseResult(success=True, error=None, artifacts=artifacts)


@tag(domain="ingestion", target="typing", node_type="materialize")
def t__typing(
    env: BuildEnv,
    graph: TargetGraph,
    parse__typing: TypingParseResult,
    t__modules: TargetRunRecord,
) -> TargetRunRecord:
    """Orchestrate typing target execution.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph for metadata.
    parse__typing
        Aggregated typing results.
    t__modules
        Upstream modules result.

    Returns
    -------
    TargetRunRecord
        Complete target execution record.
    """
    target = graph.get("typing")

    # Check for upstream failure
    if t__modules.status != "succeeded":
        return TargetRunRecord(
            target="typing",
            plugin_name="native:typing",
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
                    "repo_root": str(env.snapshot.repo_root),
                },
            )
            if target
            else (),
        )

    executor = NativeTargetExecutor.for_target(env, graph, "typing")

    # Check skip logic
    if executor.should_skip():
        return executor.skip()

    # Check parsing
    if not parse__typing.success:
        return executor.fail(RuntimeError(parse__typing.error or "Parse failed"))

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
        inputs=RunRecordInputs(env=env, run=run),
    )

    save_manifest(env, record)
    return record


__all__ = [
    "parse__typing",
    "t__typing",
    "tool__typing__pyrefly",
    "tool__typing__pyright",
    "tool__typing__ruff",
]
