"""IDE integration handlers following the unified handler pattern.

This module provides handlers for IDE helper commands using the
HandlerContext pattern for consistent resource management
and output rendering.

All handlers in this module:

1. Accept HandlerContext as their only argument
2. Return CliResult[T]
3. Never write to stdout/stderr directly
4. Never call sys.exit()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from codeintel.cli.core import CliResult
from codeintel.cli.errors import ProblemDetail
from codeintel.cli.handlers.context import HandlerContext
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource

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


def ide_hints_handler(ctx: HandlerContext) -> CliResult[IdeHintsResult]:
    """Emit IDE hints (module + subsystem context) for a relative file path.

    This handler uses the HandlerContext pattern:

    - Gateway access is lazy via ctx.gateway
    - Graph runtime access is lazy via ctx.graph_runtime
    - All resources are cleaned up by the context manager

    Parameters
    ----------
    ctx
        Handler context with:

        - params["rel_path"]: Relative path to query hints for
        - runtime: Resolved project runtime
        - gateway: Lazy storage gateway access
        - graph_runtime: Lazy graph runtime access

    Returns
    -------
    CliResult[IdeHintsResult]
        Result with hints for the file.

    Examples
    --------
    >>> with handler_context(config, runtime, {"rel_path": "pkg/mod.py"}) as ctx:
    ...     result = ide_hints_handler(ctx)  # doctest: +SKIP
    ...     result.success  # doctest: +SKIP
    True
    """
    rel_path = ctx.require_str("rel_path")

    ctx.logger.debug("Resolving IDE hints for: %s", rel_path)

    # Build backend resource using context's lazy resources
    resource = build_backend_resource(
        ctx.runtime.serving,
        gateway=ctx.gateway,
        options=BackendResourceOptions(graph_runtime=ctx.graph_runtime),
    )

    # Get hints from backend
    response = resource.backend.get_file_hints(rel_path=rel_path)

    if not response.found or not response.hints:
        LOG.debug("No IDE hints found for %s", rel_path)
        return CliResult.fail(
            ProblemDetail(
                type="codeintel:ide/hints-not-found",
                title="No hints found",
                status=404,
                detail=f"No IDE hints found for: {rel_path}",
                instance=f"file://{rel_path}",
            )
        )

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
