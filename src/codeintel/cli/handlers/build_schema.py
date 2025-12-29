"""Handlers for build schema commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args

from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.build.schemas.diff import compute_manifest_diffs
from codeintel.build.schemas.manifest import (
    ArtifactProvenance,
    ExportArtifact,
    ManifestDerivationKind,
    SchemaManifest,
    TableProvenance,
)
from codeintel.build.schemas.registry import get_schema_provider
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_execution_failed,
    fail_file_not_found,
    fail_invalid_module,
    fail_invalid_targets,
    fail_missing_required,
)
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema, normalize_column_type

if TYPE_CHECKING:
    from codeintel.build.schemas.diff import ManifestDiffResult
    from codeintel.build.schemas.manifest import ExportArtifactKind
    from codeintel.build.targets import TargetModule
    from codeintel.cli.context import CommandContext


_VALID_MODULES: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics", "export")

_ALLOWED_DERIVATION_KINDS: frozenset[str] = frozenset(get_args(ManifestDerivationKind))


@dataclass(frozen=True)
class _SchemaSelection:
    targets: tuple[str, ...] | None
    module: TargetModule | None
    all_targets: bool
    infer_native: bool
    stable: bool
    include_views: bool
    include_artifacts: bool
    include_provenance: bool


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
    try:
        return normalize_column_type(value)
    except ValueError as exc:
        msg = f"Unsupported column type: {value}"
        raise ValueError(msg) from exc


def _parse_description(value: object, *, ctx: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"Expected string or null for {ctx}"
    raise TypeError(msg)


def _parse_optional_str(value: object, *, ctx: str) -> str | None:
    """Parse an optional string field.

    Parameters
    ----------
    value
        Raw value to parse.
    ctx
        Context string for error reporting.

    Returns
    -------
    str | None
        Parsed string or None.

    Raises
    ------
    TypeError
        If the value is not a string or null.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"Expected string or null for {ctx}"
    raise TypeError(msg)


def _parse_derivation_kind(
    value: object,
    *,
    ctx: str,
) -> ManifestDerivationKind | None:
    """Parse a derivation kind value.

    Parameters
    ----------
    value
        Raw derivation kind value.
    ctx
        Context string for error reporting.

    Returns
    -------
    ManifestDerivationKind | None
        Parsed derivation kind or None when missing.

    Raises
    ------
    ValueError
        If the value is not a supported derivation kind.
    """
    raw = _parse_optional_str(value, ctx=ctx)
    if raw is None:
        return None
    if raw not in _ALLOWED_DERIVATION_KINDS:
        msg = f"Unsupported derivation kind: {raw}"
        raise ValueError(msg)
    return cast("ManifestDerivationKind", raw)


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
        infer_native=ctx.params.get_bool("infer_native"),
        stable=ctx.params.get_bool("stable"),
        include_views=ctx.params.get_bool("include_views"),
        include_artifacts=ctx.params.get_bool("include_artifacts"),
        include_provenance=ctx.params.get_bool("include_provenance"),
    )


@dataclass(frozen=True)
class _CompiledManifest:
    payload: str
    table_count: int
    view_count: int
    artifact_count: int
    manifest: SchemaManifest


def _compile_manifest(ctx: CommandContext) -> _CompiledManifest:
    selection = _parse_selection(ctx)

    request = SchemaManifestRequest(
        targets=selection.targets,
        module=selection.module,
        all_targets=selection.all_targets,
        infer_native=selection.infer_native,
        stable=selection.stable,
        include_views=selection.include_views,
        include_artifacts=selection.include_artifacts,
        include_provenance=selection.include_provenance,
    )

    runtime = ctx.runtime
    runtime_bundle = compose_cli_runtime_bundle(runtime=runtime, gateway=ctx.gateway)
    schema_index = runtime_bundle.schema_index
    if schema_index is None:
        msg = "Runtime schema_index is required to compile schema manifests"
        raise RuntimeError(msg)
    manifest = compile_schema_manifest(
        provider=get_schema_provider(),
        context=SchemaManifestContext(
            catalog=runtime_bundle.catalog,
            schema_index=schema_index,
            tag_query=runtime_bundle.tag_query,
        ),
        request=request,
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


def _parse_table_provenance(
    table_obj: dict[str, object],
    *,
    ctx: str,
) -> TableProvenance | None:
    """Parse provenance fields for a table or view.

    Parameters
    ----------
    table_obj
        JSON dictionary representing a table schema.
    ctx
        Context string for error reporting.

    Returns
    -------
    TableProvenance | None
        Parsed provenance if present.

    Raises
    ------
    ValueError
        If provenance fields are incomplete.
    """
    schema_hash = _parse_optional_str(
        table_obj.get("schema_hash"),
        ctx=f"{ctx}.schema_hash",
    )
    derivation_kind = _parse_derivation_kind(
        table_obj.get("derivation_kind"),
        ctx=f"{ctx}.derivation_kind",
    )
    derivation_source = _parse_optional_str(
        table_obj.get("derivation_source"),
        ctx=f"{ctx}.derivation_source",
    )
    producer_module = _parse_optional_str(
        table_obj.get("producer_module"),
        ctx=f"{ctx}.producer_module",
    )
    producer_version = _parse_optional_str(
        table_obj.get("producer_version"),
        ctx=f"{ctx}.producer_version",
    )
    producer_target = _parse_optional_str(
        table_obj.get("producer_target"),
        ctx=f"{ctx}.producer_target",
    )
    if schema_hash is None and derivation_kind is None and derivation_source is None:
        return None
    if schema_hash is None or derivation_kind is None or derivation_source is None:
        msg = f"Incomplete provenance fields for {ctx}"
        raise ValueError(msg)
    return TableProvenance(
        schema_hash=schema_hash,
        derivation_kind=derivation_kind,
        derivation_source=derivation_source,
        producer_module=producer_module,
        producer_version=producer_version,
        producer_target=producer_target,
    )


def _parse_artifact_provenance(
    artifact_obj: dict[str, object],
    *,
    ctx: str,
) -> ArtifactProvenance | None:
    """Parse artifact provenance metadata from JSON.

    Parameters
    ----------
    artifact_obj
        JSON dictionary representing an export artifact.
    ctx
        Context string for error reporting.

    Returns
    -------
    ArtifactProvenance | None
        Parsed provenance if present.

    Raises
    ------
    TypeError
        If provenance fields are not the expected types.
    ValueError
        If provenance fields are inconsistent.
    """
    provenance_raw = artifact_obj.get("provenance")
    if provenance_raw is None:
        return None
    if not isinstance(provenance_raw, dict):
        msg = f"Expected object for {ctx}.provenance"
        raise TypeError(msg)

    source_table_keys_raw = provenance_raw.get("source_table_keys", [])
    if not isinstance(source_table_keys_raw, list):
        msg = f"Expected list for {ctx}.provenance.source_table_keys"
        raise TypeError(msg)
    source_table_keys: list[str] = []
    for value in source_table_keys_raw:
        if not isinstance(value, str):
            msg = f"Expected string in {ctx}.provenance.source_table_keys"
            raise TypeError(msg)
        source_table_keys.append(value)

    source_schema_hashes_raw = provenance_raw.get("source_schema_hashes", [])
    if not isinstance(source_schema_hashes_raw, list):
        msg = f"Expected list for {ctx}.provenance.source_schema_hashes"
        raise TypeError(msg)
    source_schema_hashes: list[str] = []
    for value in source_schema_hashes_raw:
        if not isinstance(value, str):
            msg = f"Expected string in {ctx}.provenance.source_schema_hashes"
            raise TypeError(msg)
        source_schema_hashes.append(value)

    if len(source_schema_hashes) != len(source_table_keys):
        msg = f"Provenance length mismatch for {ctx}"
        raise ValueError(msg)

    return ArtifactProvenance(
        source_table_keys=tuple(source_table_keys),
        source_schema_hashes=tuple(source_schema_hashes),
    )


def _parse_schema_entries(
    items: list[object],
    *,
    ctx: str,
) -> tuple[list[TableSchema], dict[str, TableProvenance]]:
    """Parse table or view entries with provenance from JSON.

    Parameters
    ----------
    items
        Raw list of JSON entries to parse.
    ctx
        Context string for error reporting.

    Returns
    -------
    tuple[list[TableSchema], dict[str, TableProvenance]]
        Parsed schemas and provenance mapping keyed by table_key.
    """
    schemas: list[TableSchema] = []
    provenance_map: dict[str, TableProvenance] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        schema = _parse_table_from_json(item)
        schemas.append(schema)
        provenance = _parse_table_provenance(item, ctx=f"{ctx}[{idx}]")
        if provenance is not None:
            provenance_map[schema.table_key] = provenance
    return schemas, provenance_map


def _parse_artifacts_section(
    items: list[object],
    *,
    ctx: str,
) -> tuple[list[ExportArtifact], dict[str, ArtifactProvenance]]:
    """Parse artifact entries with provenance from JSON.

    Parameters
    ----------
    items
        Raw list of JSON artifact entries.
    ctx
        Context string for error reporting.

    Returns
    -------
    tuple[list[ExportArtifact], dict[str, ArtifactProvenance]]
        Parsed artifacts and provenance mapping keyed by filename.
    """
    artifacts: list[ExportArtifact] = []
    provenance_map: dict[str, ArtifactProvenance] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        artifact = _parse_artifact_from_json(item)
        artifacts.append(artifact)
        provenance = _parse_artifact_provenance(item, ctx=f"{ctx}[{idx}]")
        if provenance is not None:
            provenance_map[artifact.filename] = provenance
    return artifacts, provenance_map


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
        If 'tables' or 'views' is not a list.
    ValueError
        If the manifest version is unsupported.
    """
    version = str(obj.get("version", "")).strip() or "v2"
    if version != "v2":
        msg = f"Unsupported schema manifest version: {version}"
        raise ValueError(msg)

    # Parse tables
    tables_raw = obj.get("tables", [])
    if not isinstance(tables_raw, list):
        msg = "Expected 'tables' to be a list"
        raise TypeError(msg)
    tables, table_provenance = _parse_schema_entries(tables_raw, ctx="tables")

    # Parse views (v2 feature)
    views_raw = obj.get("views", [])
    if not isinstance(views_raw, list):
        msg = "Expected 'views' to be a list"
        raise TypeError(msg)
    views, view_provenance = _parse_schema_entries(views_raw, ctx="views")

    # Parse artifacts (v2 feature)
    artifacts_raw = obj.get("artifacts", [])
    if not isinstance(artifacts_raw, list):
        artifacts_raw = []
    artifacts, artifact_provenance = _parse_artifacts_section(
        artifacts_raw,
        ctx="artifacts",
    )

    return SchemaManifest(
        version=version,
        tables=tuple(tables),
        views=tuple(views),
        artifacts=tuple(artifacts),
        table_provenance=table_provenance,
        view_provenance=view_provenance,
        artifact_provenance=artifact_provenance,
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


def _try_compile_manifest(
    ctx: CommandContext,
) -> CliResult[_CompiledManifest] | _CompiledManifest:
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

    expected_manifest = _parse_manifest_from_json(expected_obj)
    diff_result = compute_manifest_diffs(expected_manifest, compiled.manifest)
    return _format_detailed_diff_result(
        diff_result,
        expected_path,
        fail_on_breaking=options.fail_on_breaking,
        fail_on_any=options.fail_on_any,
    )


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

    options = _DiffOptions(fail_on_breaking=fail_on_breaking, fail_on_any=fail_on_any)
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
