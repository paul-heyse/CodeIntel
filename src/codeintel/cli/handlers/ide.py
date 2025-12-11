"""IDE integration handlers following the unified handler pattern.

This module provides handlers for IDE helper commands using the
CommandContext pattern for consistent resource management
and output rendering.

All handlers in this module:

1. Accept CommandContext as their only argument
2. Return CliResult[T]
3. Never write to stdout/stderr directly
4. Never call sys.exit()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.analytics.runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_ide_hints_not_found
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True)
class IdeHintsResult:
    """Result from IDE hints lookup.

    Parameters
    ----------
    rel_path
        Relative path that was queried.
    hints
        List of hints for the file.
    meta
        Response metadata.
    """

    rel_path: str
    hints: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "rel_path": self.rel_path,
            "hints": self.hints,
            "meta": self.meta,
        }


# =============================================================================
# Handlers
# =============================================================================


def ide_hints_handler(ctx: CommandContext) -> CliResult[IdeHintsResult]:
    """Emit IDE hints (module + subsystem context) for a relative file path.

    This handler uses the CommandContext pattern:

    - Gateway access is lazy via ctx.storage.gateway
    - Graph runtime access is lazy via ctx.runtime.graph_runtime
    - All resources are cleaned up by the context manager

    Parameters
    ----------
    ctx
        Command context with:

        - params["rel_path"]: Relative path to query hints for
        - runtime: Resolved project runtime
        - storage: Lazy storage gateway access
        - serving: Serving invocation access

    Returns
    -------
    CliResult[IdeHintsResult]
        Result with hints for the file.

    Raises
    ------
    ValueError
        If rel_path is empty after stripping whitespace.

    Examples
    --------
    >>> with CommandContextBuilder().build() as ctx:
    ...     result = ide_hints_handler(ctx)  # doctest: +SKIP
    ...     result.success  # doctest: +SKIP
    True
    """
    rel_path = ctx.params.require_str("rel_path").strip()
    if not rel_path:
        msg = "rel_path cannot be empty"
        raise ValueError(msg)

    ctx.logger.debug("Resolving IDE hints for: %s", rel_path)

    gateway_config = getattr(ctx.gateway, "config", None)
    repo_root = ctx.params.get_path("repo_root") or Path.cwd()
    repo = getattr(gateway_config, "repo", None) or ctx.params.get_str("repo")
    if not isinstance(repo, str) or not repo:
        repo = repo_root.name
    commit = getattr(gateway_config, "commit", None) or ctx.params.get_str("commit")
    if not isinstance(commit, str) or not commit:
        commit = "HEAD"

    snapshot = (
        ctx.runtime.snapshot
        if ctx.has_runtime
        else SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
    )

    # Build graph runtime for this operation
    graph_runtime = build_graph_runtime(
        gateway=ctx.gateway,
        options=GraphRuntimeOptions(snapshot=snapshot),
    )

    serving_cfg = (
        ctx.runtime.serving
        if ctx.has_runtime
        else ServingConfig(
            repo_root=repo_root,
            repo=repo,
            commit=commit,
            db_path=getattr(gateway_config, "db_path", None),
        )
    )

    # Build backend resource using context's lazy resources
    resource = build_backend_resource(
        serving_cfg,
        gateway=ctx.gateway,
        options=BackendResourceOptions(graph_runtime=graph_runtime),
    )

    # Get hints from backend
    response = resource.backend.get_file_hints(rel_path=rel_path)

    if not response.found or not response.hints:
        LOG.debug("No IDE hints found for %s", rel_path)
        return fail_ide_hints_not_found(rel_path)

    ctx.logger.debug("Found %d hints for %s", len(response.hints), rel_path)

    return CliResult.ok(
        IdeHintsResult(
            rel_path=rel_path,
            hints=[hint.model_dump() for hint in response.hints],
            meta=response.meta.model_dump(),
        )
    )


__all__ = [
    "IdeHintsResult",
    "ide_hints_handler",
]
