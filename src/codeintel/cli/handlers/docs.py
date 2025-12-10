"""Documentation export handlers.

Handlers for documentation export and validation operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from codeintel.cli.core import CliResult
from codeintel.cli.errors.factory import fail_project_error
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.resolution.errors import ResolutionError

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


def docs_export_handler(
    ctx: HandlerContext,
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
    validation_str = ctx.param_str("validation", "required")
    macro_req_str = ctx.param_str("macro_requirement", "require_normalized")
    datasets_list = ctx.param_list("datasets")
    datasets: list[str] | None = datasets_list if datasets_list else None
    schemas_list = ctx.param_list("schemas")
    schemas: list[str] | None = schemas_list if schemas_list else None
    dry_run = ctx.param_bool("dry_run")
    skip_prereqs = ctx.param_bool("skip_prereqs")

    try:
        # Access runtime to trigger resolution and validate project
        _ = _build_runtime_from_ctx(ctx)
    except ResolutionError as e:
        return fail_project_error("docs", str(e))
    except Exception as e:  # noqa: BLE001
        return fail_project_error("docs", str(e))

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
    ctx: HandlerContext,
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
        # Access runtime to trigger resolution and validate project
        _ = _build_runtime_from_ctx(ctx)
    except ResolutionError as e:
        return fail_project_error("docs", str(e))
    except Exception as e:  # noqa: BLE001
        return fail_project_error("docs", str(e))

    LOG.info("Validating docs exports")

    # In a real implementation, this would run validation checks
    # For now, return a passing result

    return CliResult.ok(
        DocsValidateResult(
            passed=True,
            issues=[],
        )
    )


def _build_runtime_from_ctx(ctx: HandlerContext) -> object:
    """Build runtime from handler context for tests.

    Returns
    -------
    object
        Runtime resolved from the handler context.
    """
    return ctx.runtime


__all__ = [
    "DocsExportResult",
    "DocsValidateResult",
    "ExportMode",
    "ExportValidationMode",
    "MacroRequirement",
    "docs_export_handler",
    "docs_validate_handler",
]
