"""Dataset handlers.

Handlers for dataset listing, linting, validation, and management operations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

from codeintel.cli.context import CommandContext
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    DatasetDiffResult,
    DatasetLintResult,
    DatasetListResult,
    DatasetSnapshotResult,
)
from codeintel.cli.errors.results import (
    fail_file_not_found,
    fail_missing_required,
    fail_project_error,
)
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.storage.validation import collect_contract_issues

LOG = logging.getLogger(__name__)


def datasets_list_handler(
    ctx: CommandContext,
) -> CliResult[DatasetListResult | None]:
    """List datasets with capabilities and optional filters.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - category: Optional category filter.
        - include_internal: Include internal datasets.

    Returns
    -------
    CliResult[DatasetListResult]
        List of datasets.
    """
    category = ctx.params.get_str("category")
    include_internal = ctx.params.get_bool("include_internal")

    # Trigger runtime resolution to validate project exists
    try:
        _ = _build_runtime_from_ctx(ctx)
    except ResolutionError as e:
        return fail_project_error("datasets", str(e))

    LOG.info("Listing datasets (category=%s, include_internal=%s)", category, include_internal)

    contracts = get_dataset_contracts_by_table_key()

    dataset_dicts: list[dict[str, str | None]] = [
        {
            "name": contract.name,
            "table_key": contract.table_key,
            "category": None,  # Category not directly available in contracts
            "description": contract.description,
        }
        for contract in contracts.values()
    ]

    # Sort by name for consistent ordering
    dataset_dicts.sort(key=lambda d: d["name"] or "")

    return CliResult.ok(
        DatasetListResult(
            datasets=dataset_dicts,
            count=len(dataset_dicts),
        )
    )


def datasets_lint_handler(
    ctx: CommandContext,
) -> CliResult[DatasetLintResult | None]:
    """Validate dataset contract health.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.

    Returns
    -------
    CliResult[DatasetLintResult]
        Lint result with any issues found.
    """
    try:
        # Trigger runtime resolution for early error detection
        runtime = _build_runtime_from_ctx(ctx)
    except ResolutionError as e:
        return fail_project_error("datasets", str(e))
    except Exception as e:  # noqa: BLE001
        return fail_project_error("datasets", str(e))

    LOG.info("Linting datasets")

    runtime_gateway = getattr(runtime, "gateway", None)
    gateway = runtime_gateway or ctx.gateway
    issues = collect_contract_issues(gateway.con)

    passed = len(issues) == 0

    return CliResult.ok(
        DatasetLintResult(
            passed=passed,
            issue_count=len(issues),
            issues=issues,
        )
    )


def datasets_snapshot_handler(
    ctx: CommandContext,
) -> CliResult[DatasetSnapshotResult]:
    """Write current dataset specs to a JSON snapshot file.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - output: Output file path.

    Returns
    -------
    CliResult[DatasetSnapshotResult]
        Snapshot result.
    """
    output_path_str = ctx.params.get_str("output")
    if not output_path_str:
        return cast("CliResult[DatasetSnapshotResult]", fail_missing_required("output"))

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
    ctx: CommandContext,
) -> CliResult[DatasetDiffResult]:
    """Diff current dataset specs against a baseline.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - baseline_path: Path to baseline snapshot file.

    Returns
    -------
    CliResult[DatasetDiffResult]
        Diff result.
    """
    baseline_path_str = ctx.params.get_str("baseline_path")
    if not baseline_path_str:
        return cast("CliResult[DatasetDiffResult]", fail_missing_required("baseline_path"))

    baseline_path = Path(baseline_path_str)
    if not baseline_path.exists():
        return cast(
            "CliResult[DatasetDiffResult]",
            fail_file_not_found(str(baseline_path), domain="datasets"),
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


def _build_runtime_from_ctx(ctx: CommandContext) -> object:
    """Build runtime from command context for tests.

    Returns
    -------
    object
        Runtime resolved from the command context.
    """
    return ctx.runtime


__all__ = [
    "DatasetDiffResult",
    "DatasetLintResult",
    "DatasetListResult",
    "DatasetSnapshotResult",
    "datasets_diff_handler",
    "datasets_lint_handler",
    "datasets_list_handler",
    "datasets_snapshot_handler",
]
