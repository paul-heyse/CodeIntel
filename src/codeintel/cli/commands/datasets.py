"""Extended dataset management commands for the CodeIntel CLI.

This module provides Typer commands for comprehensive dataset contract
management, including validation, scaffolding, and catalog generation.

Commands
--------
- **lint**: Validate dataset contract health
- **list**: List datasets with capabilities
- **diff**: Diff current specs against a baseline
- **snapshot**: Write current dataset specs to JSON
- **conformance**: Run full conformance checks
- **generate-schemas**: Generate export JSON Schemas
- **catalog**: Generate dataset catalog
- **scaffold**: Create new dataset scaffold
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import typer

from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    OutputFormat,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    RuntimeCliOptions,
    VerboseOpt,
    build_runtime_or_exit,
    open_gateway_from_config,
    setup_logging,
)
from codeintel.cli.commands._option_shim import OptionSpec, wrap_command
from codeintel.export.validate_exports import (
    DEFAULT_SCHEMA_ROOT,
    validate_files,
)
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.datasets import DatasetRegistry, list_dataset_specs, load_dataset_registry
from codeintel.storage.datasets.catalog import (
    SamplingConfig,
    build_catalog,
    write_html_catalog,
    write_markdown_catalog,
)
from codeintel.storage.datasets.scaffold import ScaffoldOptions, scaffold_dataset
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.schema.json_schema import generate_export_schemas
from codeintel.storage.validation import collect_contract_issues
from codeintel.storage.validation.conformance import run_conformance

if TYPE_CHECKING:
    from codeintel.cli.commands._common import ProjectRuntime
    from codeintel.storage.gateway import StorageGateway


LOG = logging.getLogger(__name__)

datasets_ext_app = typer.Typer(
    name="datasets",
    help="Dataset contract management commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Enums and Option Bundles
# -----------------------------------------------------------------------------


class ExportValidationMode(Enum):
    """Validation strategy for exports."""

    REQUIRED = "required"
    SKIP = "skip"


class MacroRequirement(Enum):
    """Requirement policy for normalized macros."""

    REQUIRE_NORMALIZED = "require_normalized"
    ALLOW_PARTIAL = "allow_partial"


class OverwritePolicy(Enum):
    """Behavior when scaffold outputs already exist."""

    OVERWRITE = "overwrite"
    SKIP = "skip"
    ERROR = "error"


class DryRunMode(Enum):
    """Execution mode for dataset commands."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class SamplingMode(Enum):
    """Whether to perform sampling during validation."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class SamplingStrictness(Enum):
    """Strictness policy when sampling rows."""

    STRICT = "strict"
    LENIENT = "lenient"


class RegistryCheck(Enum):
    """Whether to validate against an existing registry."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class BootstrapSnippet(Enum):
    """Whether to emit a bootstrap snippet during scaffold."""

    EMIT = "emit"
    SKIP = "skip"


@dataclass(frozen=True)
class DatasetExportOptions:
    """Bundled options for dataset export and validation flows."""

    validation: ExportValidationMode = ExportValidationMode.REQUIRED
    macro_requirement: MacroRequirement = MacroRequirement.REQUIRE_NORMALIZED
    schemas: list[str] | None = None
    datasets: list[str] | None = None
    output_format: OutputFormat = OutputFormat.TEXT
    run_mode: DryRunMode = DryRunMode.EXECUTE


@dataclass(frozen=True)
class ExportValidationOptions:
    """Validation and macro requirement policy."""

    validation: ExportValidationMode
    macro_requirement: MacroRequirement


@dataclass(frozen=True)
class ExportSelectionOptions:
    """Dataset and schema selection options."""

    schemas: list[str] | None
    datasets: list[str] | None


@dataclass(frozen=True)
class ExportOutputOptions:
    """Output formatting and execution mode options."""

    output_format: OutputFormat
    run_mode: DryRunMode


@dataclass(frozen=True)
class DatasetScaffoldOptions:
    """Bundled options for dataset scaffolding."""

    output_dir: Path | None = None
    overwrite_policy: OverwritePolicy = OverwritePolicy.ERROR


@dataclass(frozen=True)
class ProjectSelection:
    """Project and repository selection options."""

    project_root: Path | None
    repo: str | None
    commit: str | None
    repo_root: Path | None


@dataclass(frozen=True)
class BuildSelection:
    """Build and database selection options."""

    db_path: Path | None
    build_dir: Path | None


@dataclass(frozen=True)
class RuntimeOptions:
    """Aggregated runtime selection options."""

    project: ProjectSelection
    build: BuildSelection

    def to_build_kwargs(self) -> dict[str, Path | str | None]:
        """Return keyword arguments for runtime construction.

        Returns
        -------
        dict[str, Path | str | None]
            Keyword arguments for runtime construction helpers.
        """
        return {
            "project_root": self.project.project_root,
            "repo": self.project.repo,
            "commit": self.project.commit,
            "db_path": self.build.db_path,
            "build_dir": self.build.build_dir,
            "repo_root": self.project.repo_root,
        }


@dataclass(frozen=True)
class LintOptions:
    """Options for lint command."""

    schema_dir: Path
    sampling: SamplingMode


@dataclass(frozen=True)
class ListFilters:
    """Dataset listing filters."""

    docs_view: str
    read_only: str
    max_description: int


@dataclass(frozen=True)
class DiffOptions:
    """Options for diff command."""

    baseline: Path | None
    output: Path | None
    against_ref: str | None
    baseline_path: Path


@dataclass(frozen=True)
class ConformanceOptions:
    """Options for conformance command."""

    schema_dir: Path
    sampling: SamplingMode
    sample_size: int


@dataclass(frozen=True)
class GenerateSchemasOptions:
    """Options for generate-schemas command."""

    output_dir: Path


@dataclass(frozen=True)
class CatalogOptions:
    """Options for catalog command."""

    output_dir: Path
    sample_rows_count: int
    sample_rows_strict: SamplingStrictness


@dataclass(frozen=True)
class ScaffoldMetadataOptions:
    """Metadata describing the scaffolded dataset."""

    kind: str
    table_key: str | None
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None


@dataclass(frozen=True)
class ScaffoldSchemaOptions:
    """Schema and validation settings for the scaffold."""

    schema_version: str
    validation_profile: str
    schema_id: str | None


@dataclass(frozen=True)
class ScaffoldFileOptions:
    """File naming options for scaffold outputs."""

    jsonl_filename: str | None
    parquet_filename: str | None
    stable_id: str | None


@dataclass(frozen=True)
class ScaffoldIOOptions:
    """Input/output options for scaffold generation."""

    specs_snapshot: Path
    scaffold: DatasetScaffoldOptions


@dataclass(frozen=True)
class ScaffoldBehaviorOptions:
    """Behavior toggles for scaffold generation."""

    run_mode: DryRunMode
    bootstrap: BootstrapSnippet
    registry_check: RegistryCheck


@dataclass(frozen=True)
class ScaffoldCliOptions:
    """Aggregated CLI options for scaffolding."""

    metadata: ScaffoldMetadataOptions
    schema: ScaffoldSchemaOptions
    files: ScaffoldFileOptions
    io: ScaffoldIOOptions
    behavior: ScaffoldBehaviorOptions


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

SchemaDirOpt = Annotated[
    Path,
    typer.Option(
        "--schema-dir",
        help="Directory containing export JSON Schemas.",
    ),
]

SampleRowsFlag = Annotated[
    SamplingMode,
    typer.Option(
        ...,
        "--sample-rows",
        flag_value=SamplingMode.ENABLED,
        help="Validate a sample of rows against JSON Schemas.",
        case_sensitive=False,
    ),
]

SampleSizeOpt = Annotated[
    int,
    typer.Option(
        ...,
        "--sample-size",
        help="Number of rows to sample per dataset.",
    ),
]

DocsViewFilterOpt = Annotated[
    str,
    typer.Option(
        ...,
        "--docs-view",
        help="Filter docs.* views: include, exclude, or only.",
    ),
]

ReadOnlyFilterOpt = Annotated[
    str,
    typer.Option(
        ...,
        "--read-only",
        help="Filter read-only datasets: include, exclude, or only.",
    ),
]

MaxDescriptionOpt = Annotated[
    int,
    typer.Option(
        ...,
        "--max-description",
        help="Maximum description length before truncation.",
    ),
]

BaselineOpt = Annotated[
    Path | None,
    typer.Option(
        ...,
        "--baseline",
        help="Path to JSON baseline from `codeintel datasets snapshot`.",
    ),
]

OutputOpt = Annotated[
    Path,
    typer.Option(
        ...,
        "--output",
        help="Output file path.",
    ),
]

OutputOptional = Annotated[
    Path | None,
    typer.Option(
        ...,
        "--output",
        help="Optional output file path.",
    ),
]

AgainstRefOpt = Annotated[
    str | None,
    typer.Option(
        ...,
        "--against-ref",
        help="Git ref to load baseline snapshot from.",
    ),
]

BaselinePathOpt = Annotated[
    Path,
    typer.Option(
        ...,
        "--baseline-path",
        help="Path of snapshot inside the git ref.",
    ),
]

OutputDirOpt = Annotated[
    Path,
    typer.Option(
        ...,
        "--output-dir",
        help="Directory to write output artifacts.",
    ),
]

DatasetsFilterOpt = Annotated[
    list[str] | None,
    typer.Option(
        ...,
        "--datasets",
        help="Dataset names to include (defaults to all).",
    ),
]

SampleRowsCountOpt = Annotated[
    int,
    typer.Option(
        ...,
        "--sample-rows",
        help="Number of sample rows per dataset (0 to skip).",
    ),
]

SampleRowsStrictFlag = Annotated[
    SamplingStrictness,
    typer.Option(
        ...,
        "--sample-rows-strict",
        flag_value=SamplingStrictness.STRICT,
        help="Fail if sampling cannot be performed.",
        case_sensitive=False,
    ),
]

ValidationModeOpt = Annotated[
    ExportValidationMode,
    typer.Option(
        ...,
        "--validation",
        help="Validation strategy for exports.",
        case_sensitive=False,
    ),
]

MacroRequirementOpt = Annotated[
    MacroRequirement,
    typer.Option(
        ...,
        "--macro-requirement",
        help="Requirement policy for normalized macros.",
        case_sensitive=False,
    ),
]

SchemasFilterOpt = Annotated[
    list[str] | None,
    typer.Option(
        ...,
        "--schemas",
        help="Schema names to include (repeat for multiple).",
    ),
]

OutputFormatOpt = typer.Option(
    OutputFormat.TEXT,
    "--output-format",
    help="Output format for command results.",
    case_sensitive=False,
    show_choices=True,
)

# Scaffold options
ScaffoldNameArg = Annotated[
    str,
    typer.Argument(help="Logical dataset name (e.g., my_dataset)."),
]

ScaffoldKindOpt = Annotated[
    str,
    typer.Option(
        "--kind",
        help="Dataset kind: table or view.",
    ),
]

TableKeyOpt = Annotated[
    str | None,
    typer.Option(
        "--table-key",
        help="Fully qualified table key.",
    ),
]

OwnerOpt = Annotated[
    str | None,
    typer.Option(
        "--owner",
        help="Owner/team for the dataset.",
    ),
]

FreshnessSlaOpt = Annotated[
    str | None,
    typer.Option(
        "--freshness-sla",
        help="Freshness expectation (e.g., daily, hourly).",
    ),
]

RetentionPolicyOpt = Annotated[
    str | None,
    typer.Option(
        "--retention-policy",
        help="Retention policy string (e.g., 90d).",
    ),
]

SchemaVersionOpt = Annotated[
    str,
    typer.Option(
        "--schema-version",
        help="Schema version identifier.",
    ),
]

ValidationProfileOpt = Annotated[
    str,
    typer.Option(
        "--validation-profile",
        help="Validation profile: strict or lenient.",
    ),
]

SchemaIdOpt = Annotated[
    str | None,
    typer.Option(
        "--schema-id",
        help="JSON Schema identifier.",
    ),
]

JsonlFilenameOpt = Annotated[
    str | None,
    typer.Option(
        "--jsonl-filename",
        help="Default JSONL filename.",
    ),
]

ParquetFilenameOpt = Annotated[
    str | None,
    typer.Option(
        "--parquet-filename",
        help="Default Parquet filename.",
    ),
]

StableIdOpt = Annotated[
    str | None,
    typer.Option(
        "--stable-id",
        help="Stable identifier for contract diffs.",
    ),
]

SpecsSnapshotOpt = Annotated[
    Path,
    typer.Option(
        "--specs-snapshot",
        help="Dataset specs snapshot to check for clashes.",
    ),
]

DryRunFlagOpt = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Plan without writing files.",
        is_flag=True,
    ),
]

OverwritePolicyOpt = Annotated[
    OverwritePolicy,
    typer.Option(
        ...,
        "--overwrite-policy",
        help="Behavior when scaffold outputs already exist.",
        case_sensitive=False,
    ),
]

BootstrapSnippetOpt = Annotated[
    BootstrapSnippet,
    typer.Option(
        ...,
        "--emit-bootstrap-snippet",
        flag_value=BootstrapSnippet.EMIT,
        help="Write combined bootstrap snippet.",
        case_sensitive=False,
    ),
]

RegistryCheckFlagOpt = Annotated[
    bool,
    typer.Option(
        "--check-registry",
        help="Validate against live registry for clashes.",
        is_flag=True,
    ),
]


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ELLIPSIS_LEN = 3


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _format_capabilities(caps: dict[str, bool]) -> str:
    """Return a compact capability label string.

    Parameters
    ----------
    caps
        Capability flags dictionary.

    Returns
    -------
    str
        Comma-separated labels or "-" when empty.
    """
    labels: list[str] = []
    if caps.get("docs_view"):
        labels.append("docs")
    if caps.get("read_only"):
        labels.append("ro")
    if caps.get("can_validate"):
        labels.append("validate")
    if caps.get("can_export_jsonl"):
        labels.append("jsonl")
    if caps.get("can_export_parquet"):
        labels.append("parquet")
    if caps.get("has_row_binding"):
        labels.append("binding")
    if caps.get("is_view") and not caps.get("docs_view"):
        labels.append("view")
    return ",".join(labels) if labels else "-"


def _caps_match(
    caps: dict[str, bool],
    *,
    docs_view_filter: str,
    read_only_filter: str,
) -> bool:
    """Apply docs/read-only filters to capability flags.

    Parameters
    ----------
    caps
        Capability flags.
    docs_view_filter
        Filter mode for docs views.
    read_only_filter
        Filter mode for read-only.

    Returns
    -------
    bool
        True when dataset matches both filters.
    """
    is_docs = bool(caps.get("docs_view"))
    is_read_only = bool(caps.get("read_only"))
    docs_ok = (docs_view_filter != "only" or is_docs) and (
        docs_view_filter != "exclude" or not is_docs
    )
    read_only_ok = (read_only_filter != "only" or is_read_only) and (
        read_only_filter != "exclude" or not is_read_only
    )
    return docs_ok and read_only_ok


def _truncate(text: str, limit: int) -> str:
    """Truncate string to maximum length with ellipsis.

    Parameters
    ----------
    text
        Text to truncate.
    limit
        Maximum length.

    Returns
    -------
    str
        Truncated text.
    """
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    if limit <= ELLIPSIS_LEN:
        return text[:limit]
    return text[: limit - ELLIPSIS_LEN] + "..."


def _load_specs_from_ref(
    *,
    repo_root: Path,
    ref: str,
    snapshot_path: Path,
) -> list[dict[str, object]]:
    """Load dataset specs snapshot from a git ref.

    Parameters
    ----------
    repo_root
        Repository root.
    ref
        Git reference.
    snapshot_path
        Path inside the ref.

    Returns
    -------
    list[dict[str, object]]
        Parsed specs.

    Raises
    ------
    RuntimeError
        When snapshot cannot be loaded.
    """
    target = f"{ref}:{snapshot_path.as_posix()}"
    runner = ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    result = runner.run("git", ["show", target], cwd=repo_root)
    if result.returncode != 0:
        message = f"Failed to load snapshot from {target}: {result.stderr.strip()}"
        raise RuntimeError(message)
    return json.loads(result.stdout)


def _diff_specs(
    current_specs: list[dict[str, object]],
    baseline_specs: list[dict[str, object]],
) -> tuple[list[str], list[str], list[str]]:
    """Compute added/removed/changed dataset names.

    Parameters
    ----------
    current_specs
        Current specs.
    baseline_specs
        Baseline specs.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Added, removed, changed names.
    """
    baseline_by_name: dict[str, dict[str, object]] = {
        str(spec.get("name")): spec for spec in baseline_specs
    }
    current_by_name: dict[str, dict[str, object]] = {
        str(spec.get("name")): spec for spec in current_specs
    }
    added = sorted(set(current_by_name) - set(baseline_by_name))
    removed = sorted(set(baseline_by_name) - set(current_by_name))
    changed = sorted(
        name
        for name in current_by_name
        if name in baseline_by_name and current_by_name[name] != baseline_by_name[name]
    )
    return added, removed, changed


class ScaffoldConfigError(Exception):
    """Configuration error while building scaffold options."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        """Initialize with message and exit code.

        Parameters
        ----------
        message
            Error message.
        exit_code
            Process exit code.
        """
        super().__init__(message)
        self.exit_code = exit_code


def _guard_existing_schema(
    existing_schema: Path,
    *,
    overwrite_allowed: bool,
    skip_overwrite: bool,
) -> ScaffoldConfigError | None:
    if not existing_schema.exists() or overwrite_allowed:
        return None
    exit_code = 0 if skip_overwrite else 1
    message = f"Schema already exists: {existing_schema}"
    return ScaffoldConfigError(message, exit_code=exit_code)


def _guard_snapshot_conflicts(
    name: str,
    resolved_stable_id: str,
    specs_snapshot: Path,
    *,
    overwrite_allowed: bool,
    skip_overwrite: bool,
) -> ScaffoldConfigError | None:
    if overwrite_allowed or not specs_snapshot.exists():
        return None
    try:
        specs = json.loads(specs_snapshot.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"Failed to parse specs snapshot {specs_snapshot}: {exc}"
        raise ScaffoldConfigError(message, exit_code=2) from exc
    names = {str(spec.get("name")) for spec in specs}
    stable_ids = {str(spec.get("stable_id")) for spec in specs if "stable_id" in spec}
    exit_code = 0 if skip_overwrite else 1
    if name in names:
        message = f"Dataset name already present in snapshot: {name}"
        return ScaffoldConfigError(message, exit_code=exit_code)
    if resolved_stable_id in stable_ids:
        message = f"Stable ID already present in snapshot: {resolved_stable_id}"
        return ScaffoldConfigError(message, exit_code=exit_code)
    return None


def _guard_registry_conflicts(
    name: str,
    resolved_stable_id: str,
    resolved_table_key: str,
    *,
    registry: DatasetRegistry | None,
    skip_overwrite: bool,
) -> ScaffoldConfigError | None:
    if registry is None:
        return None
    exit_code = 0 if skip_overwrite else 1
    if name in registry.by_name:
        message = f"Dataset name already present in registry: {name}"
        return ScaffoldConfigError(message, exit_code=exit_code)
    if resolved_stable_id in {ds.stable_id for ds in registry.by_name.values() if ds.stable_id}:
        message = f"Stable ID already present in registry: {resolved_stable_id}"
        return ScaffoldConfigError(message, exit_code=exit_code)
    if resolved_table_key in registry.by_table_key:
        message = f"Table key already present in registry: {resolved_table_key}"
        return ScaffoldConfigError(message, exit_code=exit_code)
    return None


def build_scaffold_options(
    name: str,
    *,
    options: ScaffoldCliOptions,
    registry: DatasetRegistry | None = None,
) -> ScaffoldOptions:
    """Construct scaffold options with guardrails.

    Parameters
    ----------
    name
        Dataset name.
    options
        Aggregated CLI options for the scaffold.
    registry
        Optional registry for validation.

    Returns
    -------
    ScaffoldOptions
        Validated options.

    Raises
    ------
    ScaffoldConfigError
        When validation fails.
    """
    metadata = options.metadata
    schema = options.schema
    files = options.files
    io_opts = options.io
    behavior = options.behavior

    resolved_table_key = (
        metadata.table_key or f"{'docs' if metadata.kind == 'view' else 'analytics'}.{name}"
    )
    resolved_schema_id = schema.schema_id or name
    resolved_stable_id = files.stable_id or name
    overwrite_policy = io_opts.scaffold.overwrite_policy
    overwrite_allowed = overwrite_policy == OverwritePolicy.OVERWRITE
    skip_overwrite = overwrite_policy == OverwritePolicy.SKIP
    existing_schema = Path("src/codeintel/config/schemas/export") / f"{resolved_schema_id}.json"

    guard_results = [
        _guard_existing_schema(
            existing_schema,
            overwrite_allowed=overwrite_allowed,
            skip_overwrite=skip_overwrite,
        ),
        _guard_snapshot_conflicts(
            name=name,
            resolved_stable_id=resolved_stable_id,
            specs_snapshot=io_opts.specs_snapshot,
            overwrite_allowed=overwrite_allowed,
            skip_overwrite=skip_overwrite,
        ),
        _guard_registry_conflicts(
            name=name,
            resolved_stable_id=resolved_stable_id,
            resolved_table_key=resolved_table_key,
            registry=registry,
            skip_overwrite=skip_overwrite,
        ),
    ]
    for guard_error in guard_results:
        if guard_error is not None:
            raise ScaffoldConfigError(str(guard_error), exit_code=guard_error.exit_code)

    return ScaffoldOptions(
        name=name,
        table_key=resolved_table_key,
        owner=metadata.owner,
        freshness_sla=metadata.freshness_sla,
        retention_policy=metadata.retention_policy,
        schema_version=schema.schema_version,
        stable_id=resolved_stable_id,
        validation_profile=cast("Literal['strict', 'lenient']", schema.validation_profile),
        jsonl_filename=files.jsonl_filename
        or (None if metadata.kind == "view" else f"{name}.jsonl"),
        parquet_filename=files.parquet_filename
        or (None if metadata.kind == "view" else f"{name}.parquet"),
        schema_id=resolved_schema_id,
        output_dir=io_opts.scaffold.output_dir or Path("build/dataset_scaffolds"),
        is_view=metadata.kind == "view",
        overwrite=overwrite_allowed,
        dry_run=behavior.run_mode == DryRunMode.DRY_RUN,
        emit_bootstrap_snippet=behavior.bootstrap == BootstrapSnippet.EMIT,
    )


# -----------------------------------------------------------------------------
# Option Builders
# -----------------------------------------------------------------------------


def _project_selection(
    project_root: Path | None,
    repo: str | None,
    commit: str | None,
    repo_root: Path | None,
) -> ProjectSelection:
    return ProjectSelection(
        project_root=project_root,
        repo=repo,
        commit=commit,
        repo_root=repo_root,
    )


def _build_selection(
    db_path: Path | None,
    build_dir: Path | None,
) -> BuildSelection:
    return BuildSelection(
        db_path=db_path,
        build_dir=build_dir,
    )


def _runtime_options(project: ProjectSelection, build: BuildSelection) -> RuntimeOptions:
    return RuntimeOptions(project=project, build=build)


def _scaffold_metadata_options(
    kind: str,
    table_key: str | None,
    owner: str | None,
    freshness_sla: str | None,
    retention_policy: str | None,
) -> ScaffoldMetadataOptions:
    return ScaffoldMetadataOptions(
        kind=kind,
        table_key=table_key,
        owner=owner,
        freshness_sla=freshness_sla,
        retention_policy=retention_policy,
    )


def _scaffold_schema_options(
    schema_version: str,
    validation_profile: str,
    schema_id: str | None,
) -> ScaffoldSchemaOptions:
    return ScaffoldSchemaOptions(
        schema_version=schema_version,
        validation_profile=validation_profile,
        schema_id=schema_id,
    )


def _scaffold_file_options(
    jsonl_filename: str | None,
    parquet_filename: str | None,
    stable_id: str | None,
) -> ScaffoldFileOptions:
    return ScaffoldFileOptions(
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        stable_id=stable_id,
    )


def _build_dataset_scaffold_options(
    output_dir: Path,
    overwrite_policy: OverwritePolicy,
) -> DatasetScaffoldOptions:
    return DatasetScaffoldOptions(
        output_dir=output_dir,
        overwrite_policy=overwrite_policy,
    )


def _scaffold_io_options(
    scaffold: DatasetScaffoldOptions,
    specs_snapshot: Path,
) -> ScaffoldIOOptions:
    return ScaffoldIOOptions(
        specs_snapshot=specs_snapshot,
        scaffold=scaffold,
    )


def _scaffold_behavior_options(
    dry_run: DryRunMode,
    bootstrap: BootstrapSnippet,
    registry_check: RegistryCheck,
) -> ScaffoldBehaviorOptions:
    return ScaffoldBehaviorOptions(
        run_mode=dry_run,
        bootstrap=bootstrap,
        registry_check=registry_check,
    )


def _scaffold_options(
    metadata: ScaffoldMetadataOptions,
    schema: ScaffoldSchemaOptions,
    files: ScaffoldFileOptions,
    io_opts: ScaffoldIOOptions,
    behavior: ScaffoldBehaviorOptions,
) -> ScaffoldCliOptions:
    return ScaffoldCliOptions(
        metadata=metadata,
        schema=schema,
        files=files,
        io=io_opts,
        behavior=behavior,
    )


# -----------------------------------------------------------------------------
def _resolve_runtime(runtime: RuntimeOptions) -> ProjectRuntime:
    cli_options = RuntimeCliOptions(
        project_root=runtime.project.project_root,
        repo=runtime.project.repo,
        commit=runtime.project.commit,
        db_path=runtime.build.db_path,
        build_dir=runtime.build.build_dir,
        repo_root=runtime.project.repo_root,
    )
    return build_runtime_or_exit(cli_options)


def _open_gateway(
    runtime: RuntimeOptions,
    *,
    read_only: bool = True,
) -> tuple[ProjectRuntime, StorageGateway]:
    runtime_state = _resolve_runtime(runtime)
    return runtime_state, open_gateway_from_config(runtime_state.cfg, read_only=read_only)


# -----------------------------------------------------------------------------
# Commands and wrappers

_RUNTIME_SPECS = [
    OptionSpec("project_root", Path | None, ProjectRootOpt),
    OptionSpec("repo", RepoOpt, None),
    OptionSpec("commit", CommitOpt, None),
    OptionSpec("repo_root", RepoRootOpt, None),
    OptionSpec("db_path", DbPathOpt, None),
    OptionSpec("build_dir", BuildDirOpt, None),
]


def _runtime_from_kwargs(cli_kwargs: Mapping[str, object]) -> RuntimeOptions:
    project_root = cast("Path | None", cli_kwargs.get("project_root"))
    repo = cast("str | None", cli_kwargs.get("repo"))
    commit = cast("str | None", cli_kwargs.get("commit"))
    repo_root = cast("Path | None", cli_kwargs.get("repo_root"))
    db_path = cast("Path | None", cli_kwargs.get("db_path"))
    build_dir = cast("Path | None", cli_kwargs.get("build_dir"))
    project = _project_selection(
        project_root,
        repo,
        commit,
        repo_root,
    )
    build = _build_selection(
        db_path,
        build_dir,
    )
    return _runtime_options(project, build)


def _verbose_from_kwargs(cli_kwargs: Mapping[str, object]) -> int:
    return int(cast("int | str | None", cli_kwargs.get("verbose", 0)) or 0)


def datasets_lint_handler(runtime: RuntimeOptions, lint: LintOptions, verbose: int) -> None:
    """Validate dataset contract health.

    Raises
    ------
    Exit
        When validation fails or sampling is requested here.
    """
    setup_logging(verbose)
    _, gateway = _open_gateway(runtime, read_only=True)
    issues = collect_contract_issues(gateway.con, schema_base_dir=lint.schema_dir)

    if lint.sampling == SamplingMode.ENABLED:
        sys.stderr.write("Row sampling requested; run `codeintel datasets conformance` instead.\n")
        raise typer.Exit(code=2)

    if issues:
        for issue in issues:
            sys.stderr.write(f"{issue}\n")
        raise typer.Exit(code=1)

    typer.secho("Dataset contract validation passed.", fg=typer.colors.GREEN)


def _bundle_lint(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    lint = LintOptions(
        schema_dir=cast(
            "Path", cli_kwargs.get("schema_dir", Path("src/codeintel/config/schemas/export"))
        ),
        sampling=cast("SamplingMode", cli_kwargs.get("sample_rows", SamplingMode.DISABLED)),
    )
    return {"runtime": runtime, "lint": lint, "verbose": _verbose_from_kwargs(cli_kwargs)}


_LINT_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("schema_dir", SchemaDirOpt, Path("src/codeintel/config/schemas/export")),
    OptionSpec("sample_rows", SampleRowsFlag, SamplingMode.DISABLED),
    OptionSpec("verbose", int, VerboseOpt),
]


datasets_lint = datasets_ext_app.command("lint")(
    wrap_command(
        datasets_lint_handler,
        _LINT_SPECS,
        bundle=_bundle_lint,
        name="datasets_lint",
    )
)


def datasets_list_handler(
    runtime: RuntimeOptions,
    filters: ListFilters,
    verbose: int,
) -> None:
    """List datasets with capabilities and optional filters."""
    setup_logging(verbose)

    _, gateway = _open_gateway(runtime, read_only=True)
    registry = load_dataset_registry(gateway.con)

    rows: list[tuple[str, str, str, str, str]] = []
    for name, ds in sorted(registry.by_name.items()):
        caps = ds.capabilities()
        if not _caps_match(
            caps,
            docs_view_filter=filters.docs_view,
            read_only_filter=filters.read_only,
        ):
            continue
        rows.append(
            (
                name,
                ds.table_key,
                ds.family or "",
                _format_capabilities(caps),
                _truncate(ds.description or "", filters.max_description),
            )
        )

    if not rows:
        sys.stdout.write("No datasets matched the requested filters.\n")
        return

    headers = ("name", "table", "family", "caps", "description")
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _fmt(row: tuple[str, ...]) -> str:
        parts: list[str] = []
        for idx, value in enumerate(row):
            if idx == len(row) - 1:
                parts.append(value)
            else:
                parts.append(value.ljust(widths[idx]))
        return "  ".join(parts)

    sys.stdout.write(_fmt(headers) + "\n")
    for row in rows:
        sys.stdout.write(_fmt(row) + "\n")


def _bundle_list(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    filters = ListFilters(
        docs_view=cast("str", cli_kwargs.get("docs_view", "include")),
        read_only=cast("str", cli_kwargs.get("read_only", "include")),
        max_description=int(cast("int | str | None", cli_kwargs.get("max_description", 80)) or 80),
    )
    return {"runtime": runtime, "filters": filters, "verbose": _verbose_from_kwargs(cli_kwargs)}


_LIST_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("docs_view", DocsViewFilterOpt, "include"),
    OptionSpec("read_only", ReadOnlyFilterOpt, "include"),
    OptionSpec("max_description", MaxDescriptionOpt, 80),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_list = datasets_ext_app.command("list")(
    wrap_command(
        datasets_list_handler,
        _LIST_SPECS,
        bundle=_bundle_list,
        name="datasets_list",
    )
)


def datasets_snapshot_handler(
    runtime: RuntimeOptions,
    output: Path,
    verbose: int,
) -> None:
    """Write current dataset specs to a JSON snapshot file."""
    setup_logging(verbose)

    _, gateway = _open_gateway(runtime, read_only=True)
    specs = list_dataset_specs(load_dataset_registry(gateway.con))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    typer.secho(f"Wrote dataset specs to {output}", fg=typer.colors.GREEN)


def _bundle_snapshot(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    return {
        "runtime": runtime,
        "output": cast("Path", cli_kwargs["output"]),
        "verbose": _verbose_from_kwargs(cli_kwargs),
    }


_SNAPSHOT_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("output", OutputOpt, OutputOpt),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_snapshot = datasets_ext_app.command("snapshot")(
    wrap_command(
        datasets_snapshot_handler,
        _SNAPSHOT_SPECS,
        bundle=_bundle_snapshot,
        name="datasets_snapshot",
    )
)


def datasets_diff_handler(
    runtime: RuntimeOptions,
    options: DiffOptions,
    verbose: int,
) -> None:
    """Diff current dataset specs against a baseline.

    Raises
    ------
    Exit
        When inputs are invalid or differences are found.
    """
    setup_logging(verbose)

    runtime_state, gateway = _open_gateway(runtime, read_only=True)
    current_specs = list_dataset_specs(load_dataset_registry(gateway.con))

    baseline_specs: list[dict[str, object]] = []
    if options.against_ref:
        baseline_specs = _load_specs_from_ref(
            repo_root=runtime_state.cfg.paths.repo_root,
            ref=options.against_ref,
            snapshot_path=options.baseline_path,
        )
    elif options.baseline is not None:
        if not options.baseline.exists():
            typer.secho(
                f"Baseline file not found: {options.baseline}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)
        baseline_specs = json.loads(options.baseline.read_text(encoding="utf-8"))
    else:
        typer.secho("Provide either --baseline or --against-ref", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    added, removed, changed = _diff_specs(current_specs, baseline_specs)

    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(current_specs, indent=2), encoding="utf-8")

    if not (added or removed or changed):
        typer.secho("No dataset spec differences detected.", fg=typer.colors.GREEN)
        return

    if added:
        sys.stdout.write(f"Added datasets: {', '.join(added)}\n")
    if removed:
        sys.stdout.write(f"Removed datasets: {', '.join(removed)}\n")
    if changed:
        sys.stdout.write(f"Changed datasets: {', '.join(changed)}\n")
    raise typer.Exit(code=1)


def _bundle_diff(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    options = DiffOptions(
        baseline=cast("Path | None", cli_kwargs.get("baseline")),
        output=cast("Path | None", cli_kwargs.get("output")),
        against_ref=cast("str | None", cli_kwargs.get("against_ref")),
        baseline_path=cast(
            "Path", cli_kwargs.get("baseline_path", Path("build/dataset_specs.json"))
        ),
    )
    return {"runtime": runtime, "options": options, "verbose": _verbose_from_kwargs(cli_kwargs)}


_DIFF_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("baseline", BaselineOpt, None),
    OptionSpec("output", OutputOptional, None),
    OptionSpec("against_ref", AgainstRefOpt, None),
    OptionSpec("baseline_path", BaselinePathOpt, Path("build/dataset_specs.json")),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_diff = datasets_ext_app.command("diff")(
    wrap_command(
        datasets_diff_handler,
        _DIFF_SPECS,
        bundle=_bundle_diff,
        name="datasets_diff",
    )
)


def datasets_conformance_handler(
    runtime: RuntimeOptions,
    options: ConformanceOptions,
    verbose: int,
) -> None:
    """Run full dataset conformance checks.

    Raises
    ------
    Exit
        When conformance fails or cannot be executed.
    """
    setup_logging(verbose)

    _, gateway = _open_gateway(runtime, read_only=True)

    try:
        report = run_conformance(
            gateway.con,
            schema_base_dir=options.schema_dir,
            sample_rows=options.sampling == SamplingMode.ENABLED,
            sample_size=options.sample_size,
        )
    except (DuckDBError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        typer.secho(f"Conformance run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if not report.ok:
        for issue in report.issues:
            prefix = issue.dataset or "global"
            sys.stderr.write(f"[{prefix}] {issue.message}\n")
        raise typer.Exit(code=1)

    typer.secho("Dataset conformance passed.", fg=typer.colors.GREEN)


def _bundle_conformance(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    options = ConformanceOptions(
        schema_dir=cast(
            "Path", cli_kwargs.get("schema_dir", Path("src/codeintel/config/schemas/export"))
        ),
        sampling=cast("SamplingMode", cli_kwargs.get("sample_rows", SamplingMode.DISABLED)),
        sample_size=int(cast("int | str | None", cli_kwargs.get("sample_size", 50)) or 50),
    )
    return {"runtime": runtime, "options": options, "verbose": _verbose_from_kwargs(cli_kwargs)}


_CONFORMANCE_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("schema_dir", SchemaDirOpt, Path("src/codeintel/config/schemas/export")),
    OptionSpec("sample_rows", SampleRowsFlag, SamplingMode.DISABLED),
    OptionSpec("sample_size", SampleSizeOpt, 50),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_conformance = datasets_ext_app.command("conformance")(
    wrap_command(
        datasets_conformance_handler,
        _CONFORMANCE_SPECS,
        bundle=_bundle_conformance,
        name="datasets_conformance",
    )
)


def datasets_generate_schemas_handler(
    runtime: RuntimeOptions,
    export: DatasetExportOptions,
    schema_opts: GenerateSchemasOptions,
    verbose: int,
) -> None:
    """Generate export JSON Schemas from TypedDict row models."""
    setup_logging(verbose)

    _, gateway = _open_gateway(runtime, read_only=True)
    registry = load_dataset_registry(gateway.con)
    include = set(export.datasets) if export.datasets else None
    written = generate_export_schemas(
        registry,
        output_dir=schema_opts.output_dir,
        include_datasets=include,
    )
    if not written:
        sys.stdout.write("No schemas generated (no matching datasets with row bindings).\n")
        return
    if export.output_format is OutputFormat.JSON:
        payload = {
            "written": [str(path) for path in written],
            "count": len(written),
            "output_dir": str(schema_opts.output_dir),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    typer.secho(f"Wrote {len(written)} schemas to {schema_opts.output_dir}", fg=typer.colors.GREEN)


def _bundle_generate_schemas(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    export = DatasetExportOptions(
        validation=cast(
            "ExportValidationMode",
            cli_kwargs.get("validation", ExportValidationMode.REQUIRED),
        ),
        macro_requirement=cast(
            "MacroRequirement",
            cli_kwargs.get("macro_requirement", MacroRequirement.REQUIRE_NORMALIZED),
        ),
        schemas=cast("list[str] | None", cli_kwargs.get("schemas")),
        datasets=cast("list[str] | None", cli_kwargs.get("datasets")),
        output_format=cast("OutputFormat", cli_kwargs.get("output_format", OutputFormat.TEXT)),
        run_mode=(
            DryRunMode.DRY_RUN if bool(cli_kwargs.get("run_mode", False)) else DryRunMode.EXECUTE
        ),
    )
    schema_opts = GenerateSchemasOptions(
        output_dir=cast(
            "Path", cli_kwargs.get("output_dir", Path("src/codeintel/config/schemas/export"))
        ),
    )
    return {
        "runtime": runtime,
        "export": export,
        "schema_opts": schema_opts,
        "verbose": _verbose_from_kwargs(cli_kwargs),
    }


_GENERATE_SCHEMAS_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("output_dir", OutputDirOpt, Path("src/codeintel/config/schemas/export")),
    OptionSpec("validation", ValidationModeOpt, ExportValidationMode.REQUIRED),
    OptionSpec("macro_requirement", MacroRequirementOpt, MacroRequirement.REQUIRE_NORMALIZED),
    OptionSpec("schemas", SchemasFilterOpt, None),
    OptionSpec("datasets", DatasetsFilterOpt, None),
    OptionSpec("output_format", OutputFormat, OutputFormatOpt),
    OptionSpec(name="run_mode", annotation=DryRunFlagOpt, default=False),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_generate_schemas = datasets_ext_app.command("generate-schemas")(
    wrap_command(
        datasets_generate_schemas_handler,
        _GENERATE_SCHEMAS_SPECS,
        bundle=_bundle_generate_schemas,
        name="datasets_generate_schemas",
    )
)


def datasets_catalog_handler(
    runtime: RuntimeOptions,
    options: CatalogOptions,
    verbose: int,
) -> None:
    """Generate Markdown/HTML dataset catalog.

    Raises
    ------
    Exit
        When required assets are missing or generation fails.
    """
    setup_logging(verbose)

    warnings_seen: set[str] = set()

    def _warn(msg: str) -> None:
        if msg in warnings_seen:
            return
        warnings_seen.add(msg)
        sys.stderr.write(msg + "\n")

    runtime_state = _resolve_runtime(runtime)

    if not runtime_state.paths.db_path.exists():
        if options.sample_rows_strict == SamplingStrictness.STRICT:
            typer.secho(
                f"Database not found at {runtime_state.paths.db_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        _warn(f"Database not found at {runtime_state.paths.db_path}; writing empty catalog.")
        options.output_dir.mkdir(parents=True, exist_ok=True)
        md = write_markdown_catalog(options.output_dir, [])
        html = write_html_catalog(options.output_dir, [])
        typer.secho(f"Wrote catalog: {md}, {html}", fg=typer.colors.GREEN)
        return

    gateway = open_gateway_from_config(runtime_state.cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)

    try:
        entries = build_catalog(
            registry,
            con=gateway.con,
            sampling=SamplingConfig(
                sample_rows=options.sample_rows_count,
                sample_rows_strict=options.sample_rows_strict == SamplingStrictness.STRICT,
            ),
            warn=_warn,
        )
    except (DuckDBError, RuntimeError) as exc:
        typer.secho(f"Failed to generate catalog: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    md = write_markdown_catalog(options.output_dir, entries)
    html = write_html_catalog(options.output_dir, entries)
    typer.secho(f"Wrote catalog: {md}, {html}", fg=typer.colors.GREEN)


def _bundle_catalog(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    options = CatalogOptions(
        output_dir=cast("Path", cli_kwargs.get("output_dir", Path("build/catalog"))),
        sample_rows_count=int(
            cast("int | str | None", cli_kwargs.get("sample_rows_count", 3)) or 3
        ),
        sample_rows_strict=cast(
            "SamplingStrictness",
            cli_kwargs.get("sample_rows_strict", SamplingStrictness.LENIENT),
        ),
    )
    return {"runtime": runtime, "options": options, "verbose": _verbose_from_kwargs(cli_kwargs)}


_CATALOG_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("output_dir", OutputDirOpt, Path("build/catalog")),
    OptionSpec("sample_rows_count", SampleRowsCountOpt, 3),
    OptionSpec("sample_rows_strict", SampleRowsStrictFlag, SamplingStrictness.LENIENT),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_catalog = datasets_ext_app.command("catalog")(
    wrap_command(
        datasets_catalog_handler,
        _CATALOG_SPECS,
        bundle=_bundle_catalog,
        name="datasets_catalog",
    )
)


def datasets_scaffold_handler(
    name: str,
    runtime: RuntimeOptions,
    options: ScaffoldCliOptions,
    verbose: int,
) -> None:
    """Create a new dataset scaffold.

    Raises
    ------
    Exit
        When conflicts are detected or validation fails.
    """
    setup_logging(verbose)

    registry: DatasetRegistry | None = None
    if options.behavior.registry_check == RegistryCheck.ENABLED:
        _, gateway = _open_gateway(runtime, read_only=True)
        registry = load_dataset_registry(gateway.con)

    try:
        opts = build_scaffold_options(
            name=name,
            options=options,
            registry=registry,
        )
    except ScaffoldConfigError as exc:
        color = typer.colors.YELLOW if exc.exit_code == 0 else typer.colors.RED
        typer.secho(str(exc), fg=color, err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    result = scaffold_dataset(opts)
    sys.stdout.write(
        "Scaffold plan:\n"
        f"  TypedDict: {result.typed_dict}\n"
        f"  Row binding snippet: {result.row_binding}\n"
        f"  JSON Schema: {result.json_schema}\n"
        f"  Metadata: {result.metadata}\n"
        f"  Bootstrap snippet: {result.bootstrap_snippet}\n"
    )
    if opts.dry_run:
        typer.secho("Dry-run only; no files were written.", fg=typer.colors.YELLOW)
    else:
        typer.secho("Scaffold created successfully.", fg=typer.colors.GREEN)


def _bundle_scaffold(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _runtime_from_kwargs(cli_kwargs)
    metadata = _scaffold_metadata_options(
        kind=cast("str", cli_kwargs.get("kind", "table")),
        table_key=cast("str | None", cli_kwargs.get("table_key")),
        owner=cast("str | None", cli_kwargs.get("owner")),
        freshness_sla=cast("str | None", cli_kwargs.get("freshness_sla")),
        retention_policy=cast("str | None", cli_kwargs.get("retention_policy")),
    )
    schema = _scaffold_schema_options(
        schema_version=cast("str", cli_kwargs.get("schema_version", "1")),
        validation_profile=cast("str", cli_kwargs.get("validation_profile", "strict")),
        schema_id=cast("str | None", cli_kwargs.get("schema_id")),
    )
    files = _scaffold_file_options(
        jsonl_filename=cast("str | None", cli_kwargs.get("jsonl_filename")),
        parquet_filename=cast("str | None", cli_kwargs.get("parquet_filename")),
        stable_id=cast("str | None", cli_kwargs.get("stable_id")),
    )
    scaffold = _build_dataset_scaffold_options(
        output_dir=cast("Path", cli_kwargs.get("output_dir", Path("build/dataset_scaffolds"))),
        overwrite_policy=cast(
            "OverwritePolicy", cli_kwargs.get("overwrite_policy", OverwritePolicy.ERROR)
        ),
    )
    io_opts = _scaffold_io_options(
        scaffold=scaffold,
        specs_snapshot=cast(
            "Path", cli_kwargs.get("specs_snapshot", Path("build/catalog/dataset_specs.json"))
        ),
    )
    behavior = _scaffold_behavior_options(
        dry_run=(
            DryRunMode.DRY_RUN if bool(cli_kwargs.get("dry_run", False)) else DryRunMode.EXECUTE
        ),
        bootstrap=cast("BootstrapSnippet", cli_kwargs.get("bootstrap", BootstrapSnippet.SKIP)),
        registry_check=(
            RegistryCheck.ENABLED
            if bool(cli_kwargs.get("registry_check", False))
            else RegistryCheck.DISABLED
        ),
    )
    options = _scaffold_options(
        metadata=metadata,
        schema=schema,
        files=files,
        io_opts=io_opts,
        behavior=behavior,
    )
    return {
        "name": cast("str", cli_kwargs["name"]),
        "runtime": runtime,
        "options": options,
        "verbose": _verbose_from_kwargs(cli_kwargs),
    }


_SCAFFOLD_SPECS = [
    *_RUNTIME_SPECS,
    OptionSpec("name", ScaffoldNameArg, ScaffoldNameArg),
    OptionSpec("kind", ScaffoldKindOpt, "table"),
    OptionSpec("table_key", TableKeyOpt, None),
    OptionSpec("owner", OwnerOpt, None),
    OptionSpec("freshness_sla", FreshnessSlaOpt, None),
    OptionSpec("retention_policy", RetentionPolicyOpt, None),
    OptionSpec("schema_version", SchemaVersionOpt, "1"),
    OptionSpec("validation_profile", ValidationProfileOpt, "strict"),
    OptionSpec("schema_id", SchemaIdOpt, None),
    OptionSpec("jsonl_filename", JsonlFilenameOpt, None),
    OptionSpec("parquet_filename", ParquetFilenameOpt, None),
    OptionSpec("stable_id", StableIdOpt, None),
    OptionSpec("output_dir", OutputDirOpt, Path("build/dataset_scaffolds")),
    OptionSpec("overwrite_policy", OverwritePolicyOpt, OverwritePolicy.ERROR),
    OptionSpec("specs_snapshot", SpecsSnapshotOpt, Path("build/catalog/dataset_specs.json")),
    OptionSpec(name="dry_run", annotation=DryRunFlagOpt, default=False),
    OptionSpec("bootstrap", BootstrapSnippetOpt, BootstrapSnippet.SKIP),
    OptionSpec(name="registry_check", annotation=RegistryCheckFlagOpt, default=False),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_scaffold = datasets_ext_app.command("scaffold")(
    wrap_command(
        datasets_scaffold_handler,
        _SCAFFOLD_SPECS,
        bundle=_bundle_scaffold,
        name="datasets_scaffold",
    )
)


SchemaNameOpt = Annotated[
    str,
    typer.Option(
        "--schema",
        help="Schema name (without .json extension).",
    ),
]

SchemaRootOpt = Annotated[
    Path | None,
    typer.Option(
        "--schema-root",
        help="Root directory containing export schemas.",
    ),
]


def datasets_validate_files_handler(
    schema: str,
    files: list[Path],
    export: DatasetExportOptions,
    schema_root: Path | None,
    verbose: int,
) -> None:
    """Validate exported JSONL/Parquet files against JSON Schemas.

    Raises
    ------
    Exit
        With the validator exit code or planned/skip exits.
    """
    setup_logging(verbose)

    root = schema_root if schema_root is not None else DEFAULT_SCHEMA_ROOT

    if export.run_mode == DryRunMode.DRY_RUN:
        payload = {
            "schema": schema,
            "files": [str(path) for path in files],
            "schema_root": str(root),
            "status": "planned",
            "validation": export.validation.value,
        }
        if export.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            sys.stdout.write(
                "Dry-run: would validate files "
                f"{', '.join(str(path) for path in files)} against {schema}.\n"
            )
        raise typer.Exit(code=0)

    if export.validation == ExportValidationMode.SKIP:
        payload = {
            "schema": schema,
            "files": [str(path) for path in files],
            "schema_root": str(root),
            "status": "skipped",
        }
        if export.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            sys.stdout.write("Validation skipped by configuration.\n")
        raise typer.Exit(code=0)

    exit_code = validate_files(schema, files, schema_root=root)
    if export.output_format is OutputFormat.JSON:
        payload = {
            "schema": schema,
            "files": [str(path) for path in files],
            "schema_root": str(root),
            "validation": export.validation.value,
            "macro_requirement": export.macro_requirement.value,
            "run_mode": export.run_mode.value,
            "status": "ok" if exit_code == 0 else "failed",
            "exit_code": exit_code,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    raise typer.Exit(code=exit_code)


def _bundle_validate_files(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    export = DatasetExportOptions(
        validation=cast(
            "ExportValidationMode",
            cli_kwargs.get("validation", ExportValidationMode.REQUIRED),
        ),
        macro_requirement=cast(
            "MacroRequirement",
            cli_kwargs.get("macro_requirement", MacroRequirement.REQUIRE_NORMALIZED),
        ),
        schemas=cast("list[str] | None", cli_kwargs.get("schemas")),
        datasets=cast("list[str] | None", cli_kwargs.get("datasets")),
        output_format=cast("OutputFormat", cli_kwargs.get("output_format", OutputFormat.TEXT)),
        run_mode=(
            DryRunMode.DRY_RUN if bool(cli_kwargs.get("run_mode", False)) else DryRunMode.EXECUTE
        ),
    )
    return {
        "schema": cast("str", cli_kwargs["schema"]),
        "files": cast("list[Path]", cli_kwargs["files"]),
        "export": export,
        "schema_root": cast("Path | None", cli_kwargs.get("schema_root")),
        "verbose": _verbose_from_kwargs(cli_kwargs),
    }


_VALIDATE_FILES_SPECS = [
    OptionSpec("schema", SchemaNameOpt, SchemaNameOpt),
    OptionSpec(
        "files",
        Annotated[list[Path], typer.Argument(help="JSONL or Parquet files to validate.")],
        None,
    ),
    OptionSpec("validation", ValidationModeOpt, ExportValidationMode.REQUIRED),
    OptionSpec("macro_requirement", MacroRequirementOpt, MacroRequirement.REQUIRE_NORMALIZED),
    OptionSpec("schemas", SchemasFilterOpt, None),
    OptionSpec("datasets", DatasetsFilterOpt, None),
    OptionSpec("output_format", OutputFormat, OutputFormatOpt),
    OptionSpec(name="run_mode", annotation=DryRunFlagOpt, default=False),
    OptionSpec("schema_root", SchemaRootOpt, None),
    OptionSpec("verbose", int, VerboseOpt),
]

datasets_validate_files = datasets_ext_app.command("validate-files")(
    wrap_command(
        datasets_validate_files_handler,
        _VALIDATE_FILES_SPECS,
        bundle=_bundle_validate_files,
        name="datasets_validate_files",
    )
)


__all__ = [
    "ScaffoldConfigError",
    "build_scaffold_options",
    "datasets_ext_app",
]
