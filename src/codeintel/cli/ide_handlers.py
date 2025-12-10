"""Typer-free handlers for IDE integration commands.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.runtime import build_graph_runtime as build_graph_runtime_internal
from codeintel.cli.cli_errors import ValidationError

# Import consolidated setup_logging from handlers.base
from codeintel.cli.handlers.base import setup_logging as _setup_logging_impl
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)
from codeintel.cli.results import CliResult
from codeintel.config.primitives import SnapshotRef
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.config.models import CodeIntelConfig

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Use consolidated setup_logging from handlers.base
setup_logging = _setup_logging_impl


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeCliOptions:
    """CLI options for runtime resolution."""

    project_root: Path | None = None


@dataclass(frozen=True)
class IdeHintsOptions:
    """Options for IDE hint resolution."""

    rel_path: str
    runtime_options: RuntimeCliOptions
    verbose: int = 0


# -----------------------------------------------------------------------------
# Runtime Utilities
# -----------------------------------------------------------------------------


def build_runtime_from_cli(options: RuntimeCliOptions) -> ProjectRuntime:
    """Build a ProjectRuntime from CLI options.

    Parameters
    ----------
    options
        CLI options containing project root.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.

    Raises
    ------
    ValidationError
        If the project cannot be resolved.
    """
    try:
        project_root = find_project_root(options.project_root)
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise ValidationError(msg) from exc
    except Exception as exc:
        msg = f"Failed to load project: {exc}"
        raise ValidationError(msg) from exc


def open_gateway_from_config(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
    """Open a StorageGateway from CodeIntelConfig.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    read_only
        Whether to open read-only.

    Returns
    -------
    StorageGateway
        Opened gateway.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = (
        StorageConfig.for_readonly(cfg.paths.db_path)
        if read_only
        else StorageConfig.for_ingest(cfg.paths.db_path)
    )
    gateway_cfg = replace(
        base_cfg,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
    )
    return open_gateway(gateway_cfg)


def build_graph_runtime(
    cfg: CodeIntelConfig,
    gateway: StorageGateway,
) -> GraphRuntime:
    """Build a graph runtime from config and gateway.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    gateway
        Storage gateway.

    Returns
    -------
    GraphRuntime
        Graph runtime instance.
    """
    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    return build_graph_runtime_internal(
        gateway,
        GraphRuntimeOptions(
            snapshot=snapshot,
            backend=cfg.graph_backend,
            features=cfg.graph_features,
        ),
    )


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Structured Handler (accepts ExecutionContext)
# -----------------------------------------------------------------------------


def ide_hints_ctx(ctx: ExecutionContext) -> CliResult[IdeHintsResult]:
    """Emit IDE hints (module + subsystem context) for a relative file path.

    Parameters
    ----------
    ctx
        Execution context with params:
        - rel_path: Relative path to query hints for.
        - project_root: Optional project root override.

    Returns
    -------
    CliResult[IdeHintsResult]
        Result with hints for the file.

    Raises
    ------
    RuntimeError
        If hints cannot be resolved.
    """
    setup_logging(ctx.verbosity)

    rel_path = ctx.require_str_param("rel_path")
    project_root_raw = ctx.params.get("project_root")
    project_root = Path(project_root_raw) if project_root_raw else None

    try:
        project_root_resolved = find_project_root(project_root)
        runtime = build_project_runtime(project_root_resolved)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise RuntimeError(msg) from exc
    except Exception as exc:
        msg = f"Failed to load project: {exc}"
        raise RuntimeError(msg) from exc

    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    graph_runtime = build_graph_runtime(runtime.cfg, gateway)

    resource = build_backend_resource(
        runtime.serving,
        gateway=gateway,
        options=BackendResourceOptions(graph_runtime=graph_runtime),
    )

    response = resource.backend.get_file_hints(rel_path=rel_path)
    if not response.found or not response.hints:
        LOG.error("No IDE hints found for %s", rel_path)
        msg = f"No hints found for: {rel_path}"
        raise RuntimeError(msg)

    return CliResult.ok(
        IdeHintsResult(
            rel_path=rel_path,
            hints=[hint.model_dump() for hint in response.hints],
            meta=response.meta.model_dump(),
        )
    )


__all__ = [
    "IdeHintsOptions",
    "IdeHintsResult",
    "RuntimeCliOptions",
    "build_graph_runtime",
    "build_runtime_from_cli",
    "ide_hints_ctx",
    "open_gateway_from_config",
    "setup_logging",
]
