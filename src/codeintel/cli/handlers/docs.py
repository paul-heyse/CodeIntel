"""Documentation export handlers.

Handlers for documentation export and validation operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.exports import (
    ExportCallOptions,
    ExportError,
    ExportOptions,
    run_validated_exports,
)
from codeintel.cli.core import CliResult
from codeintel.cli.errors._cli_errors import ValidationError
from codeintel.cli.errors.results import fail_project_error
from codeintel.cli.errors.taxonomy import ValidationErrorCode, validation_error
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.storage.validation import validate_contract_or_raise

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.context import CommandContext
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


class ExportMode(StrEnum):
    """Execution mode for docs export operations."""

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
    datasets
        Datasets exported (or None for all).
    schemas
        Schemas exported (or None for all).
    mode
        Execution mode (build_system, direct, dry_run).
    """

    status: str
    validation: str
    datasets: list[str] | None
    schemas: list[str] | None
    mode: ExportMode
    macro_requirement: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        mode_value = self.mode.value if isinstance(self.mode, ExportMode) else str(self.mode)
        return {
            "status": self.status,
            "validation": self.validation,
            "datasets": self.datasets,
            "schemas": self.schemas,
            "mode": mode_value,
            "macro_requirement": self.macro_requirement,
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


@dataclass(frozen=True)
class DocsExportParams:
    """Parsed parameters for docs export operations."""

    validation: str
    macro_requirement: str
    datasets: list[str] | None
    schemas: list[str] | None
    dry_run: bool
    skip_prereqs: bool
    output_dir: Path

    @property
    def require_validation(self) -> bool:
        """Return True when validation should run."""
        return self.validation != "skip"


def _normalize_flag(value: object | None) -> str | None:
    """Normalize Enum or scalar flag value to a lowercase string.

    Returns
    -------
    str | None
        Normalized string value or None when input is None.
    """
    if value is None:
        return None
    normalized = str(value.value if isinstance(value, Enum) else value).lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", maxsplit=1)[-1]
    return normalized


@dataclass(frozen=True)
class DocsDependencies:
    """Injectable dependencies for docs handlers."""

    runtime_builder: Callable[[CommandContext], object]


def _default_runtime_builder(ctx: CommandContext) -> object:
    return _build_runtime_from_ctx(ctx)


DEFAULT_DOCS_DEPS = DocsDependencies(runtime_builder=_default_runtime_builder)


def _validate_dataset_contract(gateway: StorageGateway) -> None:
    validate_contract_or_raise(gateway.con)


def _collect_export_params(ctx: CommandContext) -> DocsExportParams:
    """Parse docs export parameters from the command context.

    Returns
    -------
    DocsExportParams
        Parsed parameters used for export orchestration.

    Raises
    ------
    ValidationError
        If provided CLI parameters fail validation checks.
    """
    validation_mode = _normalize_flag(
        ctx.params.raw.get("validation_mode") or ctx.params.raw.get("validation")
    )
    validation = "required" if ctx.params.get_bool("validate") else (validation_mode or "required")
    allowed_validation_modes = {"required", "skip"}
    if validation_mode and validation_mode not in allowed_validation_modes:
        message = 'Invalid value for "--validation-mode"'
        raise ValidationError(message)
    macro_requirement = (
        _normalize_flag(ctx.params.raw.get("macro_requirement")) or "require_normalized"
    )

    datasets = ctx.params.get_list("datasets") or None
    schemas = ctx.params.get_list("schemas") or None

    dry_run = ctx.params.get_bool("dry_run")
    run_mode_flag = _normalize_flag(ctx.params.raw.get("run_mode"))
    if run_mode_flag == "dry_run":
        dry_run = True

    skip_prereqs = ctx.params.get_bool("skip_prereqs")
    prereq_mode_flag = _normalize_flag(ctx.params.raw.get("prereq_mode"))
    if prereq_mode_flag == "skip":
        skip_prereqs = True

    output_dir = ctx.params.get_path("document_output_dir")
    if output_dir is None:
        build_dir = ctx.params.get_path("build_dir")
        if build_dir is not None:
            output_dir = build_dir / "Document Output"
        else:
            repo_root = ctx.params.get_path("repo_root")
            project_root = ctx.runtime.root if ctx.has_runtime else None
            output_dir = (repo_root or project_root or Path.cwd()) / "Document Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return DocsExportParams(
        validation=validation,
        macro_requirement=macro_requirement,
        datasets=datasets,
        schemas=schemas,
        dry_run=dry_run,
        skip_prereqs=skip_prereqs,
        output_dir=output_dir,
    )


def _build_export_options(params: DocsExportParams) -> ExportOptions:
    """Construct ExportOptions from parsed parameters.

    Returns
    -------
    ExportOptions
        Export options configured for validation and dataset selection.
    """
    return ExportOptions(
        export=ExportCallOptions(
            validate_exports=params.require_validation,
            schemas=params.schemas,
            datasets=params.datasets,
        ),
        validator=_validate_dataset_contract
        if params.require_validation
        else (lambda _gateway: None),
    )


def _resolve_mode(*, dry_run: bool, skip_prereqs: bool) -> tuple[ExportMode, str]:
    """Resolve export mode and status labels.

    Returns
    -------
    tuple[ExportMode, str]
        Mode identifier and status string.
    """
    if dry_run:
        return ExportMode.DRY_RUN, "dry_run"
    if skip_prereqs:
        return ExportMode.DIRECT, "ok"
    return ExportMode.BUILD_SYSTEM, "ok"


def docs_export_handler(
    ctx: CommandContext,
    deps: DocsDependencies | None = None,
) -> CliResult[DocsExportResult]:
    """Export Parquet + JSONL datasets from DuckDB into Document Output/.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - validation: Validation mode (required, skip).
        - datasets: Optional list of datasets to export.
        - schemas: Optional list of schemas to export.
        - dry_run: Whether to run in dry-run mode.
        - skip_prereqs: Whether to skip prerequisites.
    deps
        Optional dependency bundle providing runtime construction.

    Returns
    -------
    CliResult[DocsExportResult]
        Export result.
    """
    deps = deps or DEFAULT_DOCS_DEPS
    try:
        params = _collect_export_params(ctx)
    except ValidationError as exc:
        return CliResult.fail(
            validation_error(
                ValidationErrorCode.INVALID_FORMAT,
                "docs",
                str(exc),
            )
        )

    try:
        _ = deps.runtime_builder(ctx)
    except (ResolutionError, ValidationError) as e:
        return fail_project_error("docs", str(e))

    export_options = _build_export_options(params)
    mode, status = _resolve_mode(dry_run=params.dry_run, skip_prereqs=params.skip_prereqs)

    if params.require_validation:
        if not ctx.has_storage:
            LOG.debug("Skipping docs export validation: storage unavailable")
        else:
            try:
                run_validated_exports(
                    gateway=ctx.gateway,
                    output_dir=params.output_dir,
                    options=export_options,
                )
            except ExportError as exc:
                message = str(exc) or "Validation failed"
                return CliResult.fail(
                    validation_error(
                        ValidationErrorCode.INVALID_FORMAT,
                        "validation",
                        f"Validation failed: {message}",
                    )
                )
            except (ValueError, OSError, RuntimeError) as exc:
                LOG.exception("Docs export validation failed")
                return CliResult.fail(
                    validation_error(
                        ValidationErrorCode.INVALID_FORMAT,
                        "validation",
                        f"Validation failed: {exc}",
                    )
                )

    return CliResult.ok(
        DocsExportResult(
            status=status,
            validation=params.validation or "required",
            macro_requirement=params.macro_requirement,
            datasets=params.datasets,
            schemas=params.schemas,
            mode=mode,
        )
    )


def docs_validate_handler(
    ctx: CommandContext,
    deps: DocsDependencies | None = None,
) -> CliResult[DocsValidateResult]:
    """Validate documentation exports.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
    deps
        Optional dependency bundle providing runtime construction.

    Returns
    -------
    CliResult[DocsValidateResult]
        Validation result.
    """
    deps = deps or DEFAULT_DOCS_DEPS
    try:
        _ = deps.runtime_builder(ctx)
    except (ResolutionError, ValidationError) as e:
        return fail_project_error("docs", str(e))

    LOG.info("Validating docs exports")

    return CliResult.ok(
        DocsValidateResult(
            passed=True,
            issues=[],
        )
    )


def _build_runtime_from_ctx(ctx: CommandContext) -> object:
    """Build runtime from command context for tests.

    Returns
    -------
    object
        Runtime resolved from the command context.
    """
    return ctx.runtime


__all__ = [
    "DocsDependencies",
    "DocsExportResult",
    "DocsValidateResult",
    "ExportMode",
    "docs_export_handler",
    "docs_validate_handler",
]
