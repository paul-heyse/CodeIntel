"""Lazy resource loading helpers for handlers.

This module provides lazy-loading helpers to avoid circular imports between
handlers/context.py and other modules. Following the pattern from
execution/_lazy_deps.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.execution.context import ExecutionContext
from codeintel.cli.resolution.runtime import RuntimeResolver

if TYPE_CHECKING:
    from codeintel.cli.resolution.types import ResolvedRuntime


def lazy_resolve_runtime(
    operation_id: str,
    params: dict[str, object],
    project_root: Path | None,
    database_path: Path | None,
) -> ResolvedRuntime:
    """Resolve runtime from handler context parameters.

    Parameters
    ----------
    operation_id
        Operation identifier.
    params
        Context parameters.
    project_root
        Optional project root path.
    database_path
        Optional database path.

    Returns
    -------
    ResolvedRuntime
        Resolved runtime.

    Notes
    -----
    Propagates ResolutionError from RuntimeResolver if runtime cannot
    be resolved (e.g., no project file and missing required params).
    """
    # Build params dict for ExecutionContext
    exec_params: dict[str, object] = dict(params)
    if project_root is not None:
        exec_params["project_root"] = project_root
    if database_path is not None:
        exec_params["db_path"] = database_path

    # Create a minimal execution context for resolution
    exec_ctx = ExecutionContext(
        operation_id=operation_id,
        params=exec_params,
    )

    return RuntimeResolver.resolve(exec_ctx)


__all__ = ["lazy_resolve_runtime"]
