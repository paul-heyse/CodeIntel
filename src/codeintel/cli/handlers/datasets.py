"""Dataset handlers.

Handlers for dataset listing, linting, validation, and management operations.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import duckdb

from codeintel.build.schemas.dataset_service import (
    DocsFilterMode,
    ReadOnlyFilterMode,
    list_datasets,
)
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    DatasetDiffResult,
    DatasetLintResult,
    DatasetListResult,
    DatasetParquetMigrationResult,
    DatasetScaffoldResult,
    DatasetSnapshotResult,
)
from codeintel.cli.errors.builder import ProblemBuilder
from codeintel.cli.errors.results import (
    fail_file_not_found,
    fail_invalid_value,
    fail_missing_required,
    fail_project_error,
)
from codeintel.cli.handlers.ops import dataset_list_handler
from codeintel.core.errors.taxonomy import OperationErrorCode
from codeintel.core.schemas.hashing import schema_digest, schema_hash
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.storage.datasets.paths import dataset_snapshot_dir
from codeintel.storage.gateway.relation import relation_from_table_key
from codeintel.storage.protocols.duckdb_relation import adapt_duckdb_relation_stream
from codeintel.storage.validation import collect_contract_issues

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.context import CommandContext
    from codeintel.core.gateway import BuildGateway
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.gateway.protocol import DuckDBRelation
    from pyarrow import RecordBatchReader

LOG = logging.getLogger(__name__)


class DatasetListFn(Protocol):
    """Callable protocol for dataset list implementations."""

    def __call__(
        self,
        *,
        docs_view: DocsFilterMode = "include",
        read_only: ReadOnlyFilterMode = "include",
    ) -> list[DatasetContract]:
        """List dataset contracts with filter controls."""
        ...


@dataclass(frozen=True)
class DatasetDependencies:
    """Injectable dependencies for dataset handlers."""

    list_datasets: DatasetListFn
    issue_collector: Callable[[Any], list[str]]


DEFAULT_DATASET_DEPS = DatasetDependencies(
    list_datasets=list_datasets,
    issue_collector=collect_contract_issues,
)


def datasets_list_handler(
    ctx: CommandContext,
    deps: DatasetDependencies | None = None,
) -> CliResult[DatasetListResult]:
    """List datasets with capabilities and optional filters.

    Parameters
    ----------
    ctx
        Command context with params:
        - docs_view: Filter for docs view datasets (include/exclude/only).
        - read_only: Filter for read-only datasets (include/exclude/only).
        - max_description: Optional description truncation length.
    deps
        Optional dependency overrides for testing.

    Returns
    -------
    CliResult[DatasetListResult]
        List of datasets.
    """
    deps = deps or DEFAULT_DATASET_DEPS
    LOG.info("Listing datasets")
    return dataset_list_handler(ctx, list_datasets_fn=deps.list_datasets)


def datasets_lint_handler(
    ctx: CommandContext,
    deps: DatasetDependencies | None = None,
) -> CliResult[DatasetLintResult | None]:
    """Validate dataset contract health.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
    deps
        Dependency overrides for runtime resolution and contract retrieval.

    Returns
    -------
    CliResult[DatasetLintResult]
        Lint result with any issues found.
    """
    deps = deps or DEFAULT_DATASET_DEPS
    try:
        gateway = ctx.gateway
    except (AttributeError, RuntimeError, ValueError) as exc:
        return fail_project_error("datasets", str(exc))

    LOG.info("Linting datasets")
    issues = deps.issue_collector(gateway.con)

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
    deps: DatasetDependencies | None = None,
) -> CliResult[DatasetSnapshotResult]:
    """Write current dataset specs to a JSON snapshot file.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - output: Output file path.
    deps
        Optional dependency overrides for testing.

    Returns
    -------
    CliResult[DatasetSnapshotResult]
        Snapshot result.
    """
    deps = deps or DEFAULT_DATASET_DEPS
    output_path_str = ctx.params.get_str("output")
    if not output_path_str:
        return cast("CliResult[DatasetSnapshotResult]", fail_missing_required("output"))

    output_path = Path(output_path_str)

    LOG.info("Writing dataset snapshot to %s", output_path)

    contracts = deps.list_datasets(docs_view="include", read_only="include")
    specs = [
        {
            "name": contract.name,
            "table_key": contract.table_key,
            "description": contract.description,
        }
        for contract in contracts
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
    deps: DatasetDependencies | None = None,
) -> CliResult[DatasetDiffResult]:
    """Diff current dataset specs against a baseline.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - baseline_path: Path to baseline snapshot file.
    deps
        Dependency overrides for runtime resolution and contract retrieval.

    Returns
    -------
    CliResult[DatasetDiffResult]
        Diff result.
    """
    deps = deps or DEFAULT_DATASET_DEPS

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

    contracts = deps.list_datasets(docs_view="include", read_only="include")
    current_names = {contract.name for contract in contracts}

    baseline_specs = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_names: set[str] = set()
    for s in baseline_specs:
        name = s.get("name")
        if isinstance(name, str):
            baseline_names.add(name)

    added = sorted(current_names - baseline_names)
    removed = sorted(baseline_names - current_names)

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


def _resolve_dataset_root_dir(ctx: CommandContext) -> Path | None:
    dataset_root = ctx.params.get_str("dataset_root_dir")
    if dataset_root:
        return Path(dataset_root)
    if ctx.has_runtime:
        return ctx.runtime.paths.dataset_root_dir
    return None


def _resolve_snapshot_id(ctx: CommandContext) -> str | None:
    snapshot_id = ctx.params.get_str("snapshot_id")
    if snapshot_id:
        return snapshot_id
    if ctx.has_runtime:
        return ctx.runtime.commit
    return None


def _resolve_table_keys(ctx: CommandContext) -> tuple[str, ...] | None:
    table_keys = ctx.params.get_list("table_keys") or None
    if table_keys is None:
        return None
    return tuple(str(key) for key in table_keys if key)


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    column_names = table_schema.column_names()
    if "repo" in column_names and "commit" in column_names:
        return ("repo", "commit")
    return ()


def _manifest_extras(table_schema: TableSchema) -> dict[str, object]:
    return {
        "table_schema": table_schema.to_json_obj(),
        "migration_source": "duckdb",
        "migrated_at": datetime.now(tz=UTC).isoformat(),
    }


def _parquet_metadata_payload(
    *,
    table_schema: TableSchema,
    schema_hash_value: str,
    schema_digest_value: str,
    partition_columns: tuple[str, ...],
    repo: str,
    commit: str,
    snapshot_id: str,
) -> dict[str, object]:
    columns_json = {col.name: col.type for col in table_schema.columns}
    nullability_json = {col.name: col.nullable for col in table_schema.columns}
    return {
        "codeintel.table_key": table_schema.table_key,
        "codeintel.domain": table_schema.schema,
        "codeintel.schema_hash": schema_hash_value,
        "codeintel.schema_digest": schema_digest_value,
        "codeintel.columns_json": columns_json,
        "codeintel.nullability_json": nullability_json,
        "codeintel.primary_keys_json": list(table_schema.primary_key),
        "codeintel.partition_columns_json": list(partition_columns),
        "codeintel.build_id": snapshot_id,
        "codeintel.repo": repo,
        "codeintel.commit": commit,
        "codeintel.snapshot_id": snapshot_id,
        "codeintel.generated_at": datetime.now(tz=UTC).isoformat(),
        "codeintel.migration_source": "duckdb",
    }


def _relation_reader(relation: DuckDBRelation) -> RecordBatchReader:
    adapter = adapt_duckdb_relation_stream(relation)
    return adapter.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)


def _migrate_dataset(
    *,
    gateway: BuildGateway,
    contract: DatasetContract,
    dataset_root_dir: Path,
    snapshot_id: str,
    repo: str,
    commit: str,
    overwrite: bool,
) -> bool:
    table_schema = contract.schema
    if table_schema is None:
        msg = f"Dataset schema missing for {contract.table_key}"
        raise ValueError(msg)
    snapshot_dir = dataset_snapshot_dir(
        dataset_root_dir,
        table_key=contract.table_key,
        snapshot_id=snapshot_id,
    )
    if snapshot_dir.exists():
        if not overwrite:
            return False
        if not snapshot_dir.is_dir():
            msg = f"Dataset snapshot path is not a directory: {snapshot_dir}"
            raise ValueError(msg)
        shutil.rmtree(snapshot_dir)
    relation = relation_from_table_key(gateway.con, contract.table_key)
    reader = _relation_reader(relation)
    schema_hash_value = schema_hash(table_schema)
    schema_digest_value = schema_digest(table_schema)
    partition_columns = _partition_columns_for_schema(table_schema)
    metadata = _parquet_metadata_payload(
        table_schema=table_schema,
        schema_hash_value=schema_hash_value,
        schema_digest_value=schema_digest_value,
        partition_columns=partition_columns,
        repo=repo,
        commit=commit,
        snapshot_id=snapshot_id,
    )
    options = ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        schema_hash=schema_hash_value,
        manifest_extras=_manifest_extras(table_schema),
        schema_metadata=metadata,
    )
    write_dataset(
        dataset_root=dataset_root_dir,
        table_key=contract.table_key,
        snapshot_id=snapshot_id,
        data=reader,
        options=options,
    )
    return True


def datasets_migrate_parquet_handler(
    ctx: CommandContext,
    deps: DatasetDependencies | None = None,
) -> CliResult[DatasetParquetMigrationResult]:
    """Materialize DuckDB-backed datasets as Parquet snapshots."""
    deps = deps or DEFAULT_DATASET_DEPS
    dataset_root_dir = _resolve_dataset_root_dir(ctx)
    if dataset_root_dir is None:
        return cast(
            "CliResult[DatasetParquetMigrationResult]",
            fail_missing_required("dataset_root_dir"),
        )
    snapshot_id = _resolve_snapshot_id(ctx)
    if snapshot_id is None:
        return cast(
            "CliResult[DatasetParquetMigrationResult]",
            fail_missing_required("snapshot_id"),
        )
    repo = ctx.runtime.repo if ctx.has_runtime else ""
    commit = snapshot_id
    table_keys = _resolve_table_keys(ctx)
    overwrite = ctx.params.get_bool("overwrite")

    try:
        gateway = ctx.gateway
    except (AttributeError, RuntimeError, ValueError) as exc:
        return fail_project_error("datasets", str(exc))

    dataset_root_dir.mkdir(parents=True, exist_ok=True)
    contracts = deps.list_datasets(docs_view="include", read_only="include")
    contracts_by_key = {contract.table_key: contract for contract in contracts}
    if table_keys is None:
        selected = sorted(contracts_by_key.values(), key=lambda contract: contract.table_key)
    else:
        missing = sorted(key for key in table_keys if key not in contracts_by_key)
        if missing:
            message = f"Unknown dataset table keys: {', '.join(missing)}"
            return cast(
                "CliResult[DatasetParquetMigrationResult]",
                fail_invalid_value("table_keys", ",".join(missing), message),
            )
        selected = [contracts_by_key[key] for key in table_keys]

    exported: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    for contract in selected:
        if contract.is_view:
            skipped.append(contract.table_key)
            continue
        try:
            did_export = _migrate_dataset(
                gateway=gateway,
                contract=contract,
                dataset_root_dir=dataset_root_dir,
                snapshot_id=snapshot_id,
                repo=repo,
                commit=commit,
                overwrite=overwrite,
            )
            if did_export:
                exported.append(contract.table_key)
            else:
                skipped.append(contract.table_key)
        except (duckdb.Error, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{contract.table_key}: {exc}")

    if errors:
        message = "Parquet migration failed for datasets: " + "; ".join(errors)
        problem = ProblemBuilder.operation(
            OperationErrorCode.DEPENDENCY_FAILED,
            "datasets.migrate_parquet",
            message,
        )
        return CliResult.fail(problem)

    return CliResult.ok(
        DatasetParquetMigrationResult(
            dataset_root_dir=str(dataset_root_dir),
            snapshot_id=snapshot_id,
            exported=sorted(exported),
            skipped=sorted(skipped),
        )
    )


def datasets_scaffold_handler(
    ctx: CommandContext,
    deps: DatasetDependencies | None = None,
) -> CliResult[DatasetScaffoldResult]:
    """Scaffold a new dataset definition.

    Parameters
    ----------
    ctx
        Command context with params:
        - name: Dataset name to scaffold.
        - registry_check: Whether to fail if dataset exists.
        - dry_run: Whether to skip writing outputs.
    deps
        Optional dependency overrides for testing.

    Returns
    -------
    CliResult[DatasetScaffoldResult]
        Scaffold status result.
    """
    deps = deps or DEFAULT_DATASET_DEPS
    name = ctx.params.require_str("name")
    registry_check = (ctx.params.get_str("registry_check") or "enabled").lower()
    dry_run = ctx.params.get_bool("dry_run")

    if registry_check not in {"enabled", "disabled"}:
        return fail_invalid_value(
            "registry_check",
            registry_check,
            'Valid values: "enabled" or "disabled".',
        )

    if registry_check == "enabled":
        contracts = deps.list_datasets(docs_view="include", read_only="include")
        known_names = {contract.table_key for contract in contracts}
        known_names.update(contract.name for contract in contracts)
        if name in known_names:
            problem = ProblemBuilder.operation(
                OperationErrorCode.ALREADY_EXISTS,
                "datasets.scaffold",
                f"Dataset '{name}' already exists in registry.",
            )
            return CliResult.fail(problem)

    return CliResult.ok(
        DatasetScaffoldResult(
            dataset=name,
            status="dry_run" if dry_run else "created",
            registry_check=registry_check,
        )
    )


__all__ = [
    "DatasetDiffResult",
    "DatasetLintResult",
    "DatasetListResult",
    "DatasetParquetMigrationResult",
    "DatasetScaffoldResult",
    "DatasetSnapshotResult",
    "datasets_diff_handler",
    "datasets_lint_handler",
    "datasets_list_handler",
    "datasets_migrate_parquet_handler",
    "datasets_scaffold_handler",
    "datasets_snapshot_handler",
]
