"""Tool execution for native Hamilton targets."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.native.tools import ToolExecutionResult

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.native.tools import ToolExecutionSpec


def execute_tool(
    spec: ToolExecutionSpec,
    env: BuildEnv,
) -> ToolExecutionResult:
    """Execute external tool and return result with ArtifactRef.

    Parameters
    ----------
    spec
        Tool execution specification.
    env
        Build environment with paths and snapshot.

    Returns
    -------
    ToolExecutionResult
        Execution result with artifact reference if successful.

    Examples
    --------
    >>> spec = ToolExecutionSpec(
    ...     tool_name="scip-python",
    ...     command_args=("index", "--output", "index.scip"),
    ...     output_path=Path("index.scip"),
    ... )
    >>> result = execute_tool(spec, env)
    >>> result.success
    True
    """
    start_time = time.perf_counter()

    exec_env: dict[str, str] | None = None
    if spec.env_vars:
        exec_env = dict(spec.env_vars)

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)

    tool_runner = env.providers.tool_runner
    if tool_runner is None:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolExecutionResult(
            success=False,
            artifact=None,
            duration_ms=duration_ms,
            stdout="",
            stderr="No tool runner configured (env.providers.tool_runner is None)",
            return_code=-1,
        )

    tool_result = asyncio.run(
        tool_runner.run(
            spec.tool_name,
            spec.command_args,
            cwd=env.snapshot.repo_root,
            timeout_ms=int(spec.timeout_seconds * 1000),
            env=exec_env,
        )
    )
    duration_ms = (time.perf_counter() - start_time) * 1000

    artifact: ArtifactRef | None = None
    if tool_result.success and spec.output_path.exists():
        artifact = ArtifactRef(
            name=spec.tool_name,
            artifact_type="file",
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            path=str(spec.output_path),
        )

    return ToolExecutionResult(
        success=tool_result.success,
        artifact=artifact,
        duration_ms=duration_ms,
        stdout=tool_result.stdout,
        stderr=tool_result.stderr,
        return_code=tool_result.returncode,
    )


__all__ = [
    "execute_tool",
]
