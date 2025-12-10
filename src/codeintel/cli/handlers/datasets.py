"""Dataset handlers.

Handlers for dataset listing, linting, validation, and management operations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.cli.core import CliResult
from codeintel.cli.errors import ProblemDetail, ValidationError
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.validation import collect_contract_issues

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext
    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetsListResult:
    """Result from datasets list operation.

    Parameters
    ----------
    datasets
        List of dataset information.
    count
        Total count of datasets.
    """

    datasets: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "datasets": self.datasets,
            "count": self.count,
        }


@dataclass(frozen=True)
class DatasetLintResult:
    """Result from datasets lint operation.

    Parameters
    ----------
    passed
        Whether validation passed.
    issue_count
        Number of issues found.
    issues
        List of issue descriptions.
    """

    passed: bool
    issue_count: int
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
            "issue_count": self.issue_count,
            "issues": self.issues,
        }


@dataclass(frozen=True)
class DatasetSnapshotResult:
    """Result from datasets snapshot operation.

    Parameters
    ----------
    output_path
        Path where snapshot was written.
    datasets_count
        Number of datasets in snapshot.
    """

    output_path: str
    datasets_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "output_path": self.output_path,
            "datasets_count": self.datasets_count,
        }


@dataclass(frozen=True)
class DatasetDiffResult:
    """Result from datasets diff operation.

    Parameters
    ----------
    added
        List of added dataset names.
    removed
        List of removed dataset names.
    changed
        List of changed dataset names.
    has_differences
        Whether any differences were found.
    """

    added: list[str]
    removed: list[str]
    changed: list[str]
    has_differences: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "has_differences": self.has_differences,
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


def datasets_list_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[DatasetsListResult]:
    """List datasets with capabilities and optional filters.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.
        - category: Optional category filter.
        - include_internal: Include internal datasets.

    Returns
    -------
    CliResult[DatasetsListResult]
        List of datasets.
    """
    category = _get_str_param(ctx, "category")
    include_internal = _get_bool_param(ctx, "include_internal")

    try:
        runtime = _build_runtime_from_ctx(ctx)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:datasets:project-error",
                title="Project Error",
                detail=str(e),
                status=400,
            )
        )

    LOG.info("Listing datasets (category=%s, include_internal=%s)", category, include_internal)

    contracts = get_dataset_contracts_by_table_key()

    dataset_dicts: list[dict[str, Any]] = [
        {
            "name": contract.name,
            "table_key": contract.table_key,
            "category": "",  # Category not directly available in contracts
            "description": contract.description or "",
        }
        for contract in contracts.values()
    ]

    # Sort by name for consistent ordering
    dataset_dicts.sort(key=lambda d: d["name"])

    runtime.gateway.close()

    return CliResult.ok(
        DatasetsListResult(
            datasets=dataset_dicts,
            count=len(dataset_dicts),
        )
    )


def datasets_lint_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[DatasetLintResult]:
    """Validate dataset contract health.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.

    Returns
    -------
    CliResult[DatasetLintResult]
        Lint result with any issues found.
    """
    try:
        runtime = _build_runtime_from_ctx(ctx)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:datasets:project-error",
                title="Project Error",
                detail=str(e),
                status=400,
            )
        )

    LOG.info("Linting datasets")

    issues = collect_contract_issues(runtime.gateway.con)
    runtime.gateway.close()

    passed = len(issues) == 0

    return CliResult.ok(
        DatasetLintResult(
            passed=passed,
            issue_count=len(issues),
            issues=issues,
        )
    )


def datasets_snapshot_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[DatasetSnapshotResult]:
    """Write current dataset specs to a JSON snapshot file.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.
        - output: Output file path.

    Returns
    -------
    CliResult[DatasetSnapshotResult]
        Snapshot result.
    """
    output_path_str = _get_str_param(ctx, "output")
    if not output_path_str:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:datasets:missing-param",
                title="Missing Parameter",
                detail="output parameter is required",
                status=400,
            )
        )

    output_path = Path(output_path_str)

    LOG.info("Writing dataset snapshot to %s", output_path)

    contracts = get_dataset_contracts_by_table_key()
    specs = [
        {
            "name": c.name,
            "table_key": c.table_key,
            "description": c.description,
        }
        for c in contracts.values()
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")

    return CliResult.ok(
        DatasetSnapshotResult(
            output_path=str(output_path),
            datasets_count=len(specs),
        )
    )


def datasets_diff_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[DatasetDiffResult]:
    """Diff current dataset specs against a baseline.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.
        - baseline_path: Path to baseline snapshot file.

    Returns
    -------
    CliResult[DatasetDiffResult]
        Diff result.
    """
    baseline_path_str = _get_str_param(ctx, "baseline_path")
    if not baseline_path_str:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:datasets:missing-param",
                title="Missing Parameter",
                detail="baseline_path parameter is required",
                status=400,
            )
        )

    baseline_path = Path(baseline_path_str)
    if not baseline_path.exists():
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:datasets:file-not-found",
                title="File Not Found",
                detail=f"Baseline file not found: {baseline_path}",
                status=404,
            )
        )

    LOG.info("Diffing datasets against %s", baseline_path)

    contracts = get_dataset_contracts_by_table_key()
    current_names = {c.name for c in contracts.values()}

    baseline_specs = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_names: set[str] = set()
    for s in baseline_specs:
        name = s.get("name")
        if isinstance(name, str):
            baseline_names.add(name)

    added = sorted(current_names - baseline_names)
    removed = sorted(baseline_names - current_names)

    # For changed, we'd need deeper comparison - simplified for now
    changed: list[str] = []

    has_differences = bool(added or removed or changed)

    return CliResult.ok(
        DatasetDiffResult(
            added=added,
            removed=removed,
            changed=changed,
            has_differences=has_differences,
        )
    )


__all__ = [
    "DatasetDiffResult",
    "DatasetLintResult",
    "DatasetSnapshotResult",
    "DatasetsListResult",
    "datasets_diff_handler",
    "datasets_lint_handler",
    "datasets_list_handler",
    "datasets_snapshot_handler",
]
