"""Handlers for build schema commands."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.build.schemas.diff import ManifestDiffResult, compute_manifest_diffs
from codeintel.build.schemas.manifest import SchemaManifest
from codeintel.build.schemas.provider_declared import declared_schema_provider
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_execution_failed,
    fail_file_not_found,
    fail_invalid_module,
    fail_invalid_targets,
    fail_missing_required,
)
from codeintel.core.schemas.primitives import Column, TableSchema

if TYPE_CHECKING:
    from codeintel.build.targets import TargetModule
    from codeintel.cli.context import CommandContext


_VALID_MODULES: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics", "export")


@dataclass(frozen=True)
class _SchemaSelection:
    targets: tuple[str, ...] | None
    module: TargetModule | None
    all_targets: bool
    only_native: bool
    infer_native: bool
    stable: bool


class _InvalidModuleError(ValueError):
    def __init__(self, module: str) -> None:
        super().__init__(module)
        self.module = module


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
    )


@dataclass(frozen=True)
class _CompiledManifest:
    payload: str
    table_count: int
    manifest: SchemaManifest


def _compile_manifest(ctx: CommandContext) -> _CompiledManifest:
    selection = _parse_selection(ctx)
    request = SchemaManifestRequest(
        targets=selection.targets,
        module=selection.module,
        all_targets=selection.all_targets,
        only_native=selection.only_native,
        infer_native=selection.infer_native,
        stable=selection.stable,
    )
    manifest = compile_schema_manifest(
        provider=declared_schema_provider(),
        request=request,
    )
    payload = json.dumps(manifest.to_json_obj(), indent=2, sort_keys=True) + "\n"
    return _CompiledManifest(payload=payload, table_count=len(manifest.tables), manifest=manifest)


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
    return Column(
        name=str(col_obj.get("name", "")),
        type=col_obj.get("type", "VARCHAR"),  # type: ignore[arg-type]
        nullable=bool(col_obj.get("nullable", True)),
        description=col_obj.get("description"),  # type: ignore[arg-type]
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
            _parse_column_from_json(col_obj)
            for col_obj in columns_raw
            if isinstance(col_obj, dict)
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
        description=table_obj.get("description"),  # type: ignore[arg-type]
    )


def _parse_manifest_from_json(obj: dict[str, object]) -> SchemaManifest:
    """Parse a SchemaManifest from a JSON object.

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
        If 'tables' is not a list.
    """
    version = str(obj.get("version", "v1"))
    tables_raw = obj.get("tables", [])
    if not isinstance(tables_raw, list):
        msg = "Expected 'tables' to be a list"
        raise TypeError(msg)

    tables = [
        _parse_table_from_json(table_obj)
        for table_obj in tables_raw
        if isinstance(table_obj, dict)
    ]

    return SchemaManifest(version=version, tables=tuple(tables))


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

    try:
        compiled = _compile_manifest(ctx)
    except _InvalidModuleError as exc:
        return fail_invalid_module(exc.module, _VALID_MODULES)
    except KeyError as exc:
        return fail_invalid_targets(str(exc))
    except (RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("build", str(exc), status=500)

    payload = compiled.payload
    if output_file and output_file != "-":
        Path(output_file).write_text(payload, encoding="utf-8")

    return CliResult.ok(payload, metadata={"table_count": compiled.table_count})


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


def _try_compile_manifest(ctx: CommandContext) -> CliResult[_CompiledManifest] | _CompiledManifest:
    """Compile manifest with error handling.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[_CompiledManifest] | _CompiledManifest
        Compiled manifest on success, or CliResult failure.
    """
    try:
        return _compile_manifest(ctx)
    except _InvalidModuleError as exc:
        return fail_invalid_module(exc.module, _VALID_MODULES)
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

    # Compile current manifest
    compile_result = _try_compile_manifest(ctx)
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

    # Compile current manifest
    compile_result = _try_compile_manifest(ctx)
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
