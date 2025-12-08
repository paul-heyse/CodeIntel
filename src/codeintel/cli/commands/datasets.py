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
    VerboseOpt,
    build_runtime_or_exit,
    open_gateway_from_config,
    setup_logging,
)
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
        SamplingMode.DISABLED,
        "--sample-rows",
        flag_value=SamplingMode.ENABLED,
        help="Validate a sample of rows against JSON Schemas.",
        case_sensitive=False,
    ),
]

SampleSizeOpt = Annotated[
    int,
    typer.Option(
        50,
        "--sample-size",
        help="Number of rows to sample per dataset.",
    ),
]

DocsViewFilterOpt = Annotated[
    str,
    typer.Option(
        "include",
        "--docs-view",
        help="Filter docs.* views: include, exclude, or only.",
    ),
]

ReadOnlyFilterOpt = Annotated[
    str,
    typer.Option(
        "include",
        "--read-only",
        help="Filter read-only datasets: include, exclude, or only.",
    ),
]

MaxDescriptionOpt = Annotated[
    int,
    typer.Option(
        80,
        "--max-description",
        help="Maximum description length before truncation.",
    ),
]

BaselineOpt = Annotated[
    Path | None,
    typer.Option(
        None,
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
        None,
        "--output",
        help="Optional output file path.",
    ),
]

AgainstRefOpt = Annotated[
    str | None,
    typer.Option(
        None,
        "--against-ref",
        help="Git ref to load baseline snapshot from.",
    ),
]

BaselinePathOpt = Annotated[
    Path,
    typer.Option(
        Path("build/dataset_specs.json"),
        "--baseline-path",
        help="Path of snapshot inside the git ref.",
    ),
]

OutputDirOpt = Annotated[
    Path,
    typer.Option(
        Path("build/catalog"),
        "--output-dir",
        help="Directory to write output artifacts.",
    ),
]

DatasetsFilterOpt = Annotated[
    list[str] | None,
    typer.Option(
        None,
        "--datasets",
        help="Dataset names to include (defaults to all).",
    ),
]

SampleRowsCountOpt = Annotated[
    int,
    typer.Option(
        3,
        "--sample-rows",
        help="Number of sample rows per dataset (0 to skip).",
    ),
]

SampleRowsStrictFlag = Annotated[
    SamplingStrictness,
    typer.Option(
        SamplingStrictness.LENIENT,
        "--sample-rows-strict",
        flag_value=SamplingStrictness.STRICT,
        help="Fail if sampling cannot be performed.",
        case_sensitive=False,
    ),
]

ValidationModeOpt = Annotated[
    ExportValidationMode,
    typer.Option(
        ExportValidationMode.REQUIRED,
        "--validation",
        help="Validation strategy for exports.",
        case_sensitive=False,
    ),
]

MacroRequirementOpt = Annotated[
    MacroRequirement,
    typer.Option(
        MacroRequirement.REQUIRE_NORMALIZED,
        "--macro-requirement",
        help="Requirement policy for normalized macros.",
        case_sensitive=False,
    ),
]

SchemasFilterOpt = Annotated[
    list[str] | None,
    typer.Option(
        None,
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

DryRunModeOpt = Annotated[
    DryRunMode,
    typer.Option(
        DryRunMode.EXECUTE,
        "--dry-run",
        flag_value=DryRunMode.DRY_RUN,
        help="Plan without writing files.",
        case_sensitive=False,
    ),
]

OverwritePolicyOpt = Annotated[
    OverwritePolicy,
    typer.Option(
        OverwritePolicy.ERROR,
        "--overwrite-policy",
        help="Behavior when scaffold outputs already exist.",
        case_sensitive=False,
    ),
]

BootstrapSnippetOpt = Annotated[
    BootstrapSnippet,
    typer.Option(
        BootstrapSnippet.SKIP,
        "--emit-bootstrap-snippet",
        flag_value=BootstrapSnippet.EMIT,
        help="Write combined bootstrap snippet.",
        case_sensitive=False,
    ),
]

RegistryCheckOpt = Annotated[
    RegistryCheck,
    typer.Option(
        RegistryCheck.DISABLED,
        "--check-registry",
        flag_value=RegistryCheck.ENABLED,
        help="Validate against live registry for clashes.",
        case_sensitive=False,
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
    return build_runtime_or_exit(
        project_root=runtime.project.project_root,
        repo=runtime.project.repo,
        commit=runtime.project.commit,
        db_path=runtime.build.db_path,
        build_dir=runtime.build.build_dir,
        repo_root=runtime.project.repo_root,
    )


def _open_gateway(
    runtime: RuntimeOptions,
    *,
    read_only: bool = True,
) -> tuple[ProjectRuntime, StorageGateway]:
    runtime_state = _resolve_runtime(runtime)
    return runtime_state, open_gateway_from_config(runtime_state.cfg, read_only=read_only)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@datasets_ext_app.command("lint")
def datasets_lint(
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    schema_dir: SchemaDirOpt = Path("src/codeintel/config/schemas/export"),
    sample_rows: SampleRowsFlag = SamplingMode.DISABLED,
    verbose: int = VerboseOpt,
) -> None:
    """Validate dataset contract health.

    Checks dataset contracts for schema consistency and integrity issues.

    Raises
    ------
    Exit
        When validation fails or row sampling is requested in this command.

    Examples
    --------
    .. code-block:: bash

        # Validate contracts
        codeintel datasets lint

        # Validate with row sampling
        codeintel datasets lint --sample-rows
    """
    setup_logging(verbose)

    lint_options = LintOptions(schema_dir=schema_dir, sampling=sample_rows)

    _, gateway = _open_gateway(runtime, read_only=True)
    issues = collect_contract_issues(gateway.con, schema_base_dir=lint_options.schema_dir)

    if lint_options.sampling == SamplingMode.ENABLED:
        sys.stderr.write("Row sampling requested; run `codeintel datasets conformance` instead.\n")
        raise typer.Exit(code=2)

    if issues:
        for issue in issues:
            sys.stderr.write(f"{issue}\n")
        raise typer.Exit(code=1)

    typer.secho("Dataset contract validation passed.", fg=typer.colors.GREEN)


@datasets_ext_app.command("list")
def datasets_list(
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    filters: Annotated[ListFilters, typer.Depends(_list_filters_dep)],
    verbose: int = VerboseOpt,
) -> None:
    """List datasets with capabilities and optional filters.

    Shows datasets with their capabilities, families, and descriptions.

    Examples
    --------
    .. code-block:: bash

        # List all datasets
        codeintel datasets list

        # Exclude docs views
        codeintel datasets list --docs-view exclude

        # Only show read-only datasets
        codeintel datasets list --read-only only
    """
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


@datasets_ext_app.command("snapshot")
def datasets_snapshot(
    output: OutputOpt,
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    verbose: int = VerboseOpt,
) -> None:
    """Write current dataset specs to a JSON snapshot file.

    Creates a JSON file with all current dataset specifications for
    use with the diff command.

    Examples
    --------
    .. code-block:: bash

        # Create snapshot
        codeintel datasets snapshot --output build/dataset_specs.json
    """
    setup_logging(verbose)

    _, gateway = _open_gateway(runtime, read_only=True)
    specs = list_dataset_specs(load_dataset_registry(gateway.con))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    typer.secho(f"Wrote dataset specs to {output}", fg=typer.colors.GREEN)


@datasets_ext_app.command("diff")
def datasets_diff(
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    options: Annotated[DiffOptions, typer.Depends(_diff_options_dep)],
    verbose: int = VerboseOpt,
) -> None:
    """Diff current dataset specs against a baseline.

    Compares current dataset specifications with a baseline snapshot
    to identify added, removed, or changed datasets.

    Raises
    ------
    Exit
        When inputs are invalid or differences are detected.

    Examples
    --------
    .. code-block:: bash

        # Diff against file
        codeintel datasets diff --baseline build/baseline.json

        # Diff against git ref
        codeintel datasets diff --against-ref main
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


@datasets_ext_app.command("conformance")
def datasets_conformance(
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    options: Annotated[ConformanceOptions, typer.Depends(_conformance_options_dep)],
    verbose: int = VerboseOpt,
) -> None:
    """Run full dataset conformance checks.

    Includes contract validation and optional row sampling against
    JSON schemas.

    Raises
    ------
    Exit
        When conformance fails or cannot be executed.

    Examples
    --------
    .. code-block:: bash

        # Run conformance checks
        codeintel datasets conformance

        # Include row sampling
        codeintel datasets conformance --sample-rows --sample-size 100
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


@datasets_ext_app.command("generate-schemas")
def datasets_generate_schemas(
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    export_options: Annotated[DatasetExportOptions, typer.Depends(_dataset_export_options_dep)],
    schema_options: Annotated[GenerateSchemasOptions, typer.Depends(_generate_schemas_options_dep)],
    verbose: int = VerboseOpt,
) -> None:
    """Generate export JSON Schemas from TypedDict row models.

    Creates JSON Schema files for datasets that have row bindings.

    Examples
    --------
    .. code-block:: bash

        # Generate all schemas
        codeintel datasets generate-schemas

        # Generate specific datasets
        codeintel datasets generate-schemas --datasets functions --datasets modules
    """
    setup_logging(verbose)

    _, gateway = _open_gateway(runtime, read_only=True)
    registry = load_dataset_registry(gateway.con)
    include = set(export_options.datasets) if export_options.datasets else None
    written = generate_export_schemas(
        registry,
        output_dir=schema_options.output_dir,
        include_datasets=include,
    )
    if not written:
        sys.stdout.write("No schemas generated (no matching datasets with row bindings).\n")
        return
    if export_options.output_format is OutputFormat.JSON:
        payload = {
            "written": [str(path) for path in written],
            "count": len(written),
            "output_dir": str(schema_options.output_dir),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    typer.secho(
        f"Wrote {len(written)} schemas to {schema_options.output_dir}", fg=typer.colors.GREEN
    )


@datasets_ext_app.command("catalog")
def datasets_catalog(
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    options: Annotated[CatalogOptions, typer.Depends(_catalog_options_dep)],
    verbose: int = VerboseOpt,
) -> None:
    """Generate Markdown/HTML dataset catalog.

    Creates documentation artifacts from the dataset registry.

    Raises
    ------
    Exit
        When required assets are missing in strict mode or catalog generation fails.

    Examples
    --------
    .. code-block:: bash

        # Generate catalog
        codeintel datasets catalog

        # Skip row sampling
        codeintel datasets catalog --sample-rows 0
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


@datasets_ext_app.command("scaffold")
def datasets_scaffold(
    name: ScaffoldNameArg,
    runtime: Annotated[RuntimeOptions, typer.Depends(_runtime_options_dep)],
    options: Annotated[ScaffoldCliOptions, typer.Depends(_scaffold_options_dep)],
    verbose: int = VerboseOpt,
) -> None:
    """Create a new dataset scaffold.

    Generates TypedDict, schema, bindings, and metadata files for a new dataset.

    Raises
    ------
    Exit
        When validation fails or conflicts are detected (exit code 0 when skipped).

    Examples
    --------
    .. code-block:: bash

        # Create basic scaffold
        codeintel datasets scaffold my_dataset

        # Create view scaffold
        codeintel datasets scaffold my_view --kind view

        # Dry run
        codeintel datasets scaffold my_dataset --dry-run
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


# -----------------------------------------------------------------------------
# validate-files command
# -----------------------------------------------------------------------------

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


@datasets_ext_app.command("validate-files")
def datasets_validate_files(
    schema: SchemaNameOpt,
    files: Annotated[list[Path], typer.Argument(help="JSONL or Parquet files to validate.")],
    export_options: Annotated[DatasetExportOptions, typer.Depends(_dataset_export_options_dep)],
    schema_root: SchemaRootOpt = None,
    verbose: int = VerboseOpt,
) -> None:
    """Validate exported JSONL/Parquet files against JSON Schemas.

    Validates one or more export files against a named JSON Schema definition.
    Supports both JSONL and Parquet file formats.

    Raises
    ------
    Exit
        With the validation exit code from the underlying validator.

    Examples
    --------
    .. code-block:: bash

        codeintel datasets validate-files --schema call_graph_edges exports/*.jsonl
        codeintel datasets validate-files --schema function_profile data.parquet
    """
    setup_logging(verbose)

    root = schema_root if schema_root is not None else DEFAULT_SCHEMA_ROOT

    if export_options.run_mode == DryRunMode.DRY_RUN:
        payload = {
            "schema": schema,
            "files": [str(path) for path in files],
            "schema_root": str(root),
            "status": "planned",
            "validation": export_options.validation.value,
        }
        if export_options.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            sys.stdout.write(
                "Dry-run: would validate files "
                f"{', '.join(str(path) for path in files)} against {schema}.\n"
            )
        raise typer.Exit(code=0)

    if export_options.validation == ExportValidationMode.SKIP:
        payload = {
            "schema": schema,
            "files": [str(path) for path in files],
            "schema_root": str(root),
            "status": "skipped",
        }
        if export_options.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            sys.stdout.write("Validation skipped by configuration.\n")
        raise typer.Exit(code=0)

    exit_code = validate_files(schema, files, schema_root=root)
    if export_options.output_format is OutputFormat.JSON:
        payload = {
            "schema": schema,
            "files": [str(path) for path in files],
            "schema_root": str(root),
            "validation": export_options.validation.value,
            "macro_requirement": export_options.macro_requirement.value,
            "run_mode": export_options.run_mode.value,
            "status": "ok" if exit_code == 0 else "failed",
            "exit_code": exit_code,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    raise typer.Exit(code=exit_code)


__all__ = [
    "ScaffoldConfigError",
    "build_scaffold_options",
    "datasets_ext_app",
]
