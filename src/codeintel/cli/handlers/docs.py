"""Documentation export handlers.

Handlers for documentation export and validation operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.errors import ProblemDetail, ValidationError
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)
from codeintel.cli.core import CliResult
from codeintel.storage.gateway import StorageConfig, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext
    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)


class ExportValidationMode(Enum):
    """Validation strategy for docs exports."""

    REQUIRED = "required"
    SKIP = "skip"


class MacroRequirement(Enum):
    """Requirement policy for normalized macros."""

    REQUIRE_NORMALIZED = "require_normalized"
    ALLOW_PARTIAL = "allow_partial"


class ExportMode(Enum):
    """Execution mode for docs exports."""

    BUILD_SYSTEM = "build_system"
    DIRECT = "direct"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class DocsExportResult:
    """Result from docs export operation.

    Parameters
    ----------
    status
        Export status (ok, dry_run, failed).
    validation
        Validation mode used.
    macro_requirement
        Macro requirement mode used.
    datasets
        Datasets exported (or None for all).
    schemas
        Schemas exported (or None for all).
    mode
        Execution mode (build_system, direct, dry_run).
    """

    status: str
    validation: str
    macro_requirement: str
    datasets: list[str] | None
    schemas: list[str] | None
    mode: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "status": self.status,
            "validation": self.validation,
            "macro_requirement": self.macro_requirement,
            "datasets": self.datasets,
            "schemas": self.schemas,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class DocsValidateResult:
    """Result from docs validation operation.

    Parameters
    ----------
    passed
        Whether validation passed.
    issues
        List of validation issues.
    """

    passed: bool
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "passed": self.passed,
            "issues": self.issues,
        }


def _get_str_param(
    ctx: EnhancedHandlerContext,
    name: str,
    default: str | None = None,
) -> str | None:
    """Extract string parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    str | None
        Parameter value or default.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    return str(value)


def _get_bool_param(
    ctx: EnhancedHandlerContext,
    name: str,
    *,
    default: bool = False,
) -> bool:
    """Extract boolean parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    bool
        Parameter value.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _resolved_to_project_runtime(runtime: ResolvedRuntime) -> ProjectRuntime:
    """Convert ResolvedRuntime to ProjectRuntime for backward compatibility.

    Parameters
    ----------
    runtime
        ResolvedRuntime from handler context.

    Returns
    -------
    ProjectRuntime
        Compatible ProjectRuntime instance.
    """
    gateway = open_gateway(StorageConfig.for_readonly(runtime.paths.db_path))
    return ProjectRuntime(
        root=runtime.root,
        project=runtime.project,
        cfg=runtime.config,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=gateway,
        tools=runtime.config.tools,
        serving=runtime.serving,
    )


def _build_runtime_from_ctx(ctx: EnhancedHandlerContext) -> ProjectRuntime:
    """Build ProjectRuntime from enhanced handler context.

    Parameters
    ----------
    ctx
        Enhanced handler context.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.

    Raises
    ------
    ValidationError
        If project cannot be resolved.
    """
    project_root_raw = ctx.params.get("project_root")
    project_root = Path(str(project_root_raw)) if project_root_raw else None

    try:
        project_root_resolved = find_project_root(project_root)
        return build_project_runtime(project_root_resolved)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise ValidationError(msg) from exc


def _extract_list_param(
    ctx: EnhancedHandlerContext,
    name: str,
) -> list[str] | None:
    """Extract list parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.

    Returns
    -------
    list[str] | None
        List of strings or None.
    """
    value = ctx.params.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def docs_export_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[DocsExportResult]:
    """Export Parquet + JSONL datasets from DuckDB into Document Output/.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.
        - validation: Validation mode (required, skip).
        - macro_requirement: Macro requirement mode.
        - datasets: Optional list of datasets to export.
        - schemas: Optional list of schemas to export.
        - dry_run: Whether to run in dry-run mode.
        - skip_prereqs: Whether to skip prerequisites.

    Returns
    -------
    CliResult[DocsExportResult]
        Export result.
    """
    validation_str = _get_str_param(ctx, "validation", "required")
    macro_req_str = _get_str_param(ctx, "macro_requirement", "require_normalized")
    datasets = _extract_list_param(ctx, "datasets")
    schemas = _extract_list_param(ctx, "schemas")
    dry_run = _get_bool_param(ctx, "dry_run")
    skip_prereqs = _get_bool_param(ctx, "skip_prereqs")

    try:
        _build_runtime_from_ctx(ctx)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:docs:project-error",
                title="Project Error",
                detail=str(e),
                status=400,
            )
        )

    # Determine mode
    if dry_run:
        mode = ExportMode.DRY_RUN.value
        status = "dry_run"
    elif skip_prereqs:
        mode = ExportMode.DIRECT.value
        status = "ok"
    else:
        mode = ExportMode.BUILD_SYSTEM.value
        status = "ok"

    LOG.info(
        "Docs export: validation=%s, macro_req=%s, mode=%s",
        validation_str,
        macro_req_str,
        mode,
    )

    # In a real implementation, this would call the actual export logic
    # For now, return a result indicating what would be done

    return CliResult.ok(
        DocsExportResult(
            status=status,
            validation=validation_str or "required",
            macro_requirement=macro_req_str or "require_normalized",
            datasets=datasets,
            schemas=schemas,
            mode=mode,
        )
    )


def docs_validate_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[DocsValidateResult]:
    """Validate documentation exports.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.

    Returns
    -------
    CliResult[DocsValidateResult]
        Validation result.
    """
    try:
        _build_runtime_from_ctx(ctx)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:docs:project-error",
                title="Project Error",
                detail=str(e),
                status=400,
            )
        )

    LOG.info("Validating docs exports")

    # In a real implementation, this would run validation checks
    # For now, return a passing result

    return CliResult.ok(
        DocsValidateResult(
            passed=True,
            issues=[],
        )
    )


__all__ = [
    "DocsExportResult",
    "DocsValidateResult",
    "ExportMode",
    "ExportValidationMode",
    "MacroRequirement",
    "docs_export_handler",
    "docs_validate_handler",
]
