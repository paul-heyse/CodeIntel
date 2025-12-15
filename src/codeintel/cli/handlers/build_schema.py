"""Handlers for build schema commands."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args

from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.build.schemas.diff import compute_manifest_diffs
from codeintel.build.schemas.manifest import ExportArtifact, SchemaManifest
from codeintel.build.schemas.provider_declared import declared_schema_provider
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_execution_failed,
    fail_file_not_found,
    fail_invalid_module,
    fail_invalid_targets,
    fail_missing_required,
)
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema

if TYPE_CHECKING:
    from codeintel.build.schemas.diff import ManifestDiffResult
    from codeintel.build.schemas.manifest import ExportArtifactKind
    from codeintel.build.targets import TargetModule
    from codeintel.cli.context import CommandContext
    from codeintel.storage.gateway.protocol import DuckDBConnection


_VALID_MODULES: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics", "export")

_ALLOWED_COLUMN_TYPES: frozenset[str] = frozenset(get_args(ColumnType))


@dataclass(frozen=True)
class _SchemaSelection:
    targets: tuple[str, ...] | None
    module: TargetModule | None
    all_targets: bool
    only_native: bool
    infer_native: bool
    stable: bool
    include_views: bool
    include_artifacts: bool


class _InvalidModuleError(ValueError):
    def __init__(self, module: str) -> None:
        super().__init__(module)
        self.module = module


def _parse_column_type(value: object) -> ColumnType:
    """Parse a ColumnType from an arbitrary input value.

    Parameters
    ----------
    value
        Raw value to parse.

    Returns
    -------
    ColumnType
        Parsed column type.

    Raises
    ------
    TypeError
        If value is not a string.
    ValueError
        If value is not a supported ColumnType literal.
    """
    if not isinstance(value, str):
        msg = "Expected string for column type"
        raise TypeError(msg)
    if value not in _ALLOWED_COLUMN_TYPES:
        msg = f"Unsupported column type: {value}"
        raise ValueError(msg)
    return cast("ColumnType", value)


def _parse_description(value: object, *, ctx: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"Expected string or null for {ctx}"
    raise TypeError(msg)


def _parse_module(module_raw: str | None) -> TargetModule | None:
    if module_raw is None:
        return None
    if module_raw not in _VALID_MODULES:
        raise _InvalidModuleError(module_raw)
    return cast("TargetModule", module_raw)


def _parse_selection(ctx: CommandContext) -> _SchemaSelection:
    targets_list = ctx.params.get_list("targets")
    targets: tuple[str, ...] | None = tuple(targets_list) if targets_list else None
    module = _parse_module(ctx.params.get_str("module"))
    return _SchemaSelection(
        targets=targets,
        module=module,
        all_targets=ctx.params.get_bool("all_targets"),
        only_native=ctx.params.get_bool("only_native"),
        infer_native=ctx.params.get_bool("infer_native"),
        stable=ctx.params.get_bool("stable"),
        include_views=ctx.params.get_bool("include_views"),
        include_artifacts=ctx.params.get_bool("include_artifacts"),
    )


@dataclass(frozen=True)
class _CompiledManifest:
    payload: str
    table_count: int
    view_count: int
    artifact_count: int
    manifest: SchemaManifest


class _ViewsRequireGatewayError(ValueError):
    """Raised when --include-views is specified but gateway is unavailable."""


def _compile_manifest(
    ctx: CommandContext,
    *,
    gateway_con: DuckDBConnection | None = None,
) -> _CompiledManifest:
    selection = _parse_selection(ctx)

    # Validate that --include-views has a connection available
    if selection.include_views and gateway_con is None:
        raise _ViewsRequireGatewayError

    request = SchemaManifestRequest(
        targets=selection.targets,
        module=selection.module,
        all_targets=selection.all_targets,
        only_native=selection.only_native,
        infer_native=selection.infer_native,
        stable=selection.stable,
        include_views=selection.include_views,
        include_artifacts=selection.include_artifacts,
    )

    manifest = compile_schema_manifest(
        provider=declared_schema_provider(),
        request=request,
        con=gateway_con,
    )
    payload = json.dumps(manifest.to_json_obj(), indent=2, sort_keys=True) + "\n"
    return _CompiledManifest(
        payload=payload,
        table_count=len(manifest.tables),
        view_count=len(manifest.views),
        artifact_count=len(manifest.artifacts),
        manifest=manifest,
    )


def _parse_column_from_json(col_obj: dict[str, object]) -> Column:
    """Parse a Column from a JSON object.

    Parameters
    ----------
    col_obj
        JSON dictionary representing a column.

    Returns
    -------
    Column
        Parsed column instance.
    """
    column_type = _parse_column_type(col_obj.get("type", "VARCHAR"))
    return Column(
        name=str(col_obj.get("name", "")),
        type=column_type,
        nullable=bool(col_obj.get("nullable", True)),
        description=_parse_description(col_obj.get("description"), ctx="column.description"),
    )


def _parse_table_from_json(table_obj: dict[str, object]) -> TableSchema:
    """Parse a TableSchema from a JSON object.

    Parameters
    ----------
    table_obj
        JSON dictionary representing a table schema.

    Returns
    -------
    TableSchema
        Parsed table schema instance.
    """
    columns: list[Column] = []
    columns_raw = table_obj.get("columns", [])
    if isinstance(columns_raw, list):
        columns.extend(
            _parse_column_from_json(col_obj) for col_obj in columns_raw if isinstance(col_obj, dict)
        )

    primary_key_raw = table_obj.get("primary_key", [])
    primary_key = (
        tuple(str(k) for k in primary_key_raw) if isinstance(primary_key_raw, list) else ()
    )

    return TableSchema(
        schema=str(table_obj.get("schema", "")),
        name=str(table_obj.get("name", "")),
        columns=columns,
        primary_key=primary_key,
        description=_parse_description(table_obj.get("description"), ctx="table.description"),
    )


def _parse_artifact_kind(kind_raw: object) -> ExportArtifactKind:
    """Parse an ExportArtifactKind from a raw value.

    Parameters
    ----------
    kind_raw
        Raw kind value from JSON.

    Returns
    -------
    ExportArtifactKind
        Validated artifact kind, defaults to 'jsonl' if invalid.
    """
    if kind_raw == "parquet":
        return "parquet"
    if kind_raw == "json":
        return "json"
    if kind_raw == "csv":
        return "csv"
    return "jsonl"


def _parse_artifact_from_json(artifact_obj: dict[str, object]) -> ExportArtifact:
    """Parse an ExportArtifact from a JSON object.

    Parameters
    ----------
    artifact_obj
        JSON dictionary representing an export artifact.

    Returns
    -------
    ExportArtifact
        Parsed artifact instance.
    """
    kind = _parse_artifact_kind(artifact_obj.get("kind"))

    table_key_raw = artifact_obj.get("table_key")
    table_key = str(table_key_raw) if table_key_raw is not None else None

    return ExportArtifact(
        kind=kind,
        filename=str(artifact_obj.get("filename", "")),
        table_key=table_key,
        description=_parse_description(artifact_obj.get("description"), ctx="artifact.description"),
    )


def _parse_manifest_from_json(obj: dict[str, object]) -> SchemaManifest:
    """Parse a SchemaManifest from a JSON object.

    Supports both v1 (tables only) and v2 (tables, views, artifacts) formats.

    Parameters
    ----------
    obj
        JSON object representing a schema manifest.

    Returns
    -------
    SchemaManifest
        Parsed manifest instance.

    Raises
    ------
    TypeError
        If 'tables' or 'views' is not a list.
    """
    version = str(obj.get("version", "v1"))

    # Parse tables
    tables_raw = obj.get("tables", [])
    if not isinstance(tables_raw, list):
        msg = "Expected 'tables' to be a list"
        raise TypeError(msg)
    tables = [
        _parse_table_from_json(table_obj) for table_obj in tables_raw if isinstance(table_obj, dict)
    ]

    # Parse views (v2 feature)
    views_raw = obj.get("views", [])
    if not isinstance(views_raw, list):
        msg = "Expected 'views' to be a list"
        raise TypeError(msg)
    views = [
        _parse_table_from_json(view_obj) for view_obj in views_raw if isinstance(view_obj, dict)
    ]

    # Parse artifacts (v2 feature)
    artifacts_raw = obj.get("artifacts", [])
    if not isinstance(artifacts_raw, list):
        artifacts_raw = []
    artifacts = [
        _parse_artifact_from_json(artifact_obj)
        for artifact_obj in artifacts_raw
        if isinstance(artifact_obj, dict)
    ]

    return SchemaManifest(
        version=version,
        tables=tuple(tables),
        views=tuple(views),
        artifacts=tuple(artifacts),
    )


def build_schema_compile_handler(ctx: CommandContext) -> CliResult[str]:
    """Compile a schema manifest for a target selection.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        JSON schema manifest payload.
    """
    output_format = ctx.params.get_str("output_format") or "json"
    output_file = ctx.params.get_str("output_file")

    if output_format != "json":
        return fail_execution_failed(
            "build",
            f"Unsupported schema manifest format: {output_format}",
            status=400,
        )

    # Get gateway connection if available (needed for --include-views)
    gateway_con = ctx.gateway.con if ctx.has_storage else None

    try:
        compiled = _compile_manifest(ctx, gateway_con=gateway_con)
    except _InvalidModuleError as exc:
        return fail_invalid_module(exc.module, _VALID_MODULES)
    except _ViewsRequireGatewayError:
        return fail_execution_failed(
            "build",
            "--include-views requires a database connection. "
            "Use --db-path or ensure a database is configured.",
            status=400,
        )
    except KeyError as exc:
        return fail_invalid_targets(str(exc))
    except (RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("build", str(exc), status=500)

    payload = compiled.payload
    if output_file and output_file != "-":
        Path(output_file).write_text(payload, encoding="utf-8")

    metadata: dict[str, object] = {"table_count": compiled.table_count}
    if compiled.view_count > 0:
        metadata["view_count"] = compiled.view_count
    if compiled.artifact_count > 0:
        metadata["artifact_count"] = compiled.artifact_count

    return CliResult.ok(payload, metadata=metadata)


def _load_expected_manifest(
    expected_path: Path,
) -> CliResult[dict[str, object]] | dict[str, object]:
    """Load and validate expected manifest JSON file.

    Parameters
    ----------
    expected_path
        Path to the expected manifest file.

    Returns
    -------
    CliResult[dict[str, object]] | dict[str, object]
        The parsed dict on success, or a CliResult failure on error.
    """
    try:
        expected_obj = json.loads(expected_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return fail_execution_failed(
            "build",
            f"Expected manifest is not valid JSON: {exc}",
            status=400,
        )
    if not isinstance(expected_obj, dict):
        return fail_execution_failed(
            "build",
            "Expected manifest must be a JSON object",
            status=400,
        )
    return expected_obj


def _format_detailed_diff_result(
    diff_result: ManifestDiffResult,
    expected_path: Path,
    *,
    fail_on_breaking: bool,
    fail_on_any: bool,
) -> CliResult[str]:
    """Format a detailed diff result into a CliResult.

    Parameters
    ----------
    diff_result
        The manifest diff result to format.
    expected_path
        Path to the expected manifest file.
    fail_on_breaking
        Whether to fail on breaking changes.
    fail_on_any
        Whether to fail on any changes.

    Returns
    -------
    CliResult[str]
        Formatted result with success or failure status.
    """
    summary = diff_result.format_summary()

    # Determine if we should fail based on flags
    should_fail = (fail_on_any and diff_result.has_any_changes) or (
        fail_on_breaking and diff_result.has_breaking_changes
    )

    if should_fail:
        message = (
            f"{summary}\n\n"
            "Update the expected manifest by re-running:\n"
            f"  codeintel build schema compile --output {expected_path}\n"
        )
        return fail_execution_failed("build", message, status=409)
    return CliResult.ok(summary)


def _format_legacy_diff(
    expected_json: str,
    actual_json: str,
    expected_path: Path,
) -> CliResult[str]:
    """Format a legacy string-based diff result.

    Parameters
    ----------
    expected_json
        Expected manifest JSON string.
    actual_json
        Actual manifest JSON string.
    expected_path
        Path to the expected manifest file.

    Returns
    -------
    CliResult[str]
        Failure result with diff details.
    """
    diff = "".join(
        difflib.unified_diff(
            expected_json.splitlines(keepends=True),
            actual_json.splitlines(keepends=True),
            fromfile=f"expected:{expected_path}",
            tofile="actual",
        )
    )
    message = (
        "Schema drift detected (expected vs actual).\n\n"
        f"{diff}\n"
        "Update the expected manifest by re-running:\n"
        f"  codeintel build schema compile --output {expected_path}\n"
    )
    return fail_execution_failed("build", message, status=409)


def _try_compile_manifest(
    ctx: CommandContext,
    *,
    gateway_con: DuckDBConnection | None = None,
) -> CliResult[_CompiledManifest] | _CompiledManifest:
    """Compile manifest with error handling.

    Parameters
    ----------
    ctx
        Command context.
    gateway_con
        Optional DuckDB connection for view inference.

    Returns
    -------
    CliResult[_CompiledManifest] | _CompiledManifest
        Compiled manifest on success, or CliResult failure.
    """
    try:
        return _compile_manifest(ctx, gateway_con=gateway_con)
    except _InvalidModuleError as exc:
        return fail_invalid_module(exc.module, _VALID_MODULES)
    except _ViewsRequireGatewayError:
        return fail_execution_failed(
            "build",
            "--include-views requires a database connection. "
            "Use --db-path or ensure a database is configured.",
            status=400,
        )
    except KeyError as exc:
        return fail_invalid_targets(str(exc))
    except (RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("build", str(exc), status=500)


@dataclass(frozen=True)
class _DiffOptions:
    """Options for schema diff comparison."""

    detailed: bool
    fail_on_breaking: bool
    fail_on_any: bool


def _compare_and_format_diff(
    expected_obj: dict[str, object],
    compiled: _CompiledManifest,
    expected_path: Path,
    options: _DiffOptions,
) -> CliResult[str]:
    """Compare manifests and format the result.

    Parameters
    ----------
    expected_obj
        Expected manifest as parsed JSON.
    compiled
        Compiled current manifest.
    expected_path
        Path to expected manifest file.
    options
        Diff comparison options.

    Returns
    -------
    CliResult[str]
        Comparison result.
    """
    expected_json = json.dumps(expected_obj, indent=2, sort_keys=True) + "\n"
    if expected_json == compiled.payload:
        return CliResult.ok("Schema manifest matches expected.\n")

    if options.detailed:
        expected_manifest = _parse_manifest_from_json(expected_obj)
        diff_result = compute_manifest_diffs(expected_manifest, compiled.manifest)
        return _format_detailed_diff_result(
            diff_result,
            expected_path,
            fail_on_breaking=options.fail_on_breaking,
            fail_on_any=options.fail_on_any,
        )

    return _format_legacy_diff(expected_json, compiled.payload, expected_path)


def build_schema_diff_handler(ctx: CommandContext) -> CliResult[str]:
    """Compare a compiled schema manifest to an expected manifest file.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        Success message when manifests match; otherwise a drift diff failure.
    """
    expected_file = ctx.params.get_str("expected_file")
    if not expected_file:
        return fail_missing_required("expected")

    expected_path = Path(expected_file)
    if not expected_path.exists():
        return fail_file_not_found(str(expected_path), domain="build")

    # Parse flags
    detailed = ctx.params.get_bool("detailed")
    fail_on_breaking = ctx.params.get_bool("fail_on_breaking") or True
    fail_on_any = ctx.params.get_bool("fail_on_any")

    # Load expected manifest
    load_result = _load_expected_manifest(expected_path)
    if isinstance(load_result, CliResult):
        return cast("CliResult[str]", load_result)

    # Get gateway connection if available (needed for --include-views)
    gateway_con = ctx.gateway.con if ctx.has_storage else None

    # Compile current manifest
    compile_result = _try_compile_manifest(ctx, gateway_con=gateway_con)
    if isinstance(compile_result, CliResult):
        return cast("CliResult[str]", compile_result)

    options = _DiffOptions(
        detailed=detailed,
        fail_on_breaking=fail_on_breaking,
        fail_on_any=fail_on_any,
    )
    return _compare_and_format_diff(load_result, compile_result, expected_path, options)


def _format_migration_result(
    expected_manifest: SchemaManifest | None,
    compiled: _CompiledManifest,
    expected_path: Path,
    *,
    dry_run: bool,
) -> CliResult[str]:
    """Format the migration result.

    Parameters
    ----------
    expected_manifest
        The existing manifest, or None if creating new.
    compiled
        The compiled current manifest.
    expected_path
        Path to write the manifest.
    dry_run
        Whether to perform a dry run without writing.

    Returns
    -------
    CliResult[str]
        Formatted migration result.
    """
    lines: list[str] = []

    if expected_manifest is not None:
        diff_result = compute_manifest_diffs(expected_manifest, compiled.manifest)
        if diff_result.has_any_changes:
            lines.append("Migration plan:")
            lines.append("")
            lines.append(diff_result.format_summary())
        else:
            lines.append("No changes detected. Manifest is up to date.")
    else:
        lines.append(f"Creating new manifest with {compiled.table_count} table(s).")

    if dry_run:
        lines.append("")
        lines.append("[DRY RUN] No changes written.")
        lines.append(f"Run with --no-dry-run to write to {expected_path}")
    else:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(compiled.payload, encoding="utf-8")
        lines.append("")
        lines.append(f"Manifest written to {expected_path}")

    return CliResult.ok("\n".join(lines) + "\n")


def build_schema_migrate_handler(ctx: CommandContext) -> CliResult[str]:
    """Update expected manifest to match current schemas.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        Migration result showing changes made or planned.
    """
    expected_file = ctx.params.get_str("expected_file")
    if not expected_file:
        return fail_missing_required("expected")

    expected_path = Path(expected_file)
    dry_run = ctx.params.get_bool("dry_run") or True

    # Load existing manifest if present
    expected_manifest: SchemaManifest | None = None
    if expected_path.exists():
        try:
            expected_obj = json.loads(expected_path.read_text(encoding="utf-8"))
            if isinstance(expected_obj, dict):
                expected_manifest = _parse_manifest_from_json(expected_obj)
        except (json.JSONDecodeError, TypeError):
            pass

    # Get gateway connection if available (needed for --include-views)
    gateway_con = ctx.gateway.con if ctx.has_storage else None

    # Compile current manifest
    compile_result = _try_compile_manifest(ctx, gateway_con=gateway_con)
    if isinstance(compile_result, CliResult):
        return cast("CliResult[str]", compile_result)

    return _format_migration_result(
        expected_manifest, compile_result, expected_path, dry_run=dry_run
    )


__all__ = [
    "build_schema_compile_handler",
    "build_schema_diff_handler",
    "build_schema_migrate_handler",
]
