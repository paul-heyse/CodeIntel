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
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    VerboseOpt,
    build_runtime_or_exit,
    open_gateway_from_config,
    setup_logging,
)
from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
from codeintel.storage.catalog import (
    SamplingConfig,
    build_catalog,
    write_html_catalog,
    write_markdown_catalog,
)
from codeintel.storage.conformance import run_conformance
from codeintel.storage.contract_validation import collect_contract_issues
from codeintel.storage.datasets import DatasetRegistry, list_dataset_specs, load_dataset_registry
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.scaffold import ScaffoldOptions, scaffold_dataset
from codeintel.storage.schema_generation import generate_export_schemas

LOG = logging.getLogger(__name__)

datasets_ext_app = typer.Typer(
    name="datasets",
    help="Dataset contract management commands.",
    no_args_is_help=True,
)


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
    bool,
    typer.Option(
        "--sample-rows",
        is_flag=True,
        help="Validate a sample of rows against JSON Schemas.",
    ),
]

SampleSizeOpt = Annotated[
    int,
    typer.Option(
        "--sample-size",
        help="Number of rows to sample per dataset.",
    ),
]

DocsViewFilterOpt = Annotated[
    str,
    typer.Option(
        "--docs-view",
        help="Filter docs.* views: include, exclude, or only.",
    ),
]

ReadOnlyFilterOpt = Annotated[
    str,
    typer.Option(
        "--read-only",
        help="Filter read-only datasets: include, exclude, or only.",
    ),
]

MaxDescriptionOpt = Annotated[
    int,
    typer.Option(
        "--max-description",
        help="Maximum description length before truncation.",
    ),
]

BaselineOpt = Annotated[
    Path | None,
    typer.Option(
        "--baseline",
        help="Path to JSON baseline from `codeintel datasets snapshot`.",
    ),
]

OutputOpt = Annotated[
    Path,
    typer.Option(
        "--output",
        help="Output file path.",
    ),
]

OutputOptional = Annotated[
    Path | None,
    typer.Option(
        "--output",
        help="Optional output file path.",
    ),
]

AgainstRefOpt = Annotated[
    str | None,
    typer.Option(
        "--against-ref",
        help="Git ref to load baseline snapshot from.",
    ),
]

BaselinePathOpt = Annotated[
    Path,
    typer.Option(
        "--baseline-path",
        help="Path of snapshot inside the git ref.",
    ),
]

OutputDirOpt = Annotated[
    Path,
    typer.Option(
        "--output-dir",
        help="Directory to write output artifacts.",
    ),
]

DatasetsFilterOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--datasets",
        help="Dataset names to include (defaults to all).",
    ),
]

SampleRowsCountOpt = Annotated[
    int,
    typer.Option(
        "--sample-rows",
        help="Number of sample rows per dataset (0 to skip).",
    ),
]

SampleRowsStrictFlag = Annotated[
    bool,
    typer.Option(
        "--sample-rows-strict",
        is_flag=True,
        help="Fail if sampling cannot be performed.",
    ),
]

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

DryRunFlag = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        is_flag=True,
        help="Plan without writing files.",
    ),
]

OverwriteFlag = Annotated[
    bool,
    typer.Option(
        "--overwrite",
        is_flag=True,
        help="Allow overwriting existing files.",
    ),
]

EmitBootstrapFlag = Annotated[
    bool,
    typer.Option(
        "--emit-bootstrap-snippet",
        is_flag=True,
        help="Write combined bootstrap snippet.",
    ),
]

CheckRegistryFlag = Annotated[
    bool,
    typer.Option(
        "--check-registry",
        is_flag=True,
        help="Validate against live registry for clashes.",
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
    if limit <= 0 or len(text) <= limit:
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


def build_scaffold_options(
    name: str,
    kind: str,
    table_key: str | None,
    owner: str | None,
    freshness_sla: str | None,
    retention_policy: str | None,
    schema_version: str,
    validation_profile: str,
    schema_id: str | None,
    jsonl_filename: str | None,
    parquet_filename: str | None,
    stable_id: str | None,
    specs_snapshot: Path,
    output_dir: Path,
    overwrite: bool,
    dry_run: bool,
    emit_bootstrap_snippet: bool,
    registry: DatasetRegistry | None = None,
) -> ScaffoldOptions:
    """Construct scaffold options with guardrails.

    Parameters
    ----------
    name
        Dataset name.
    kind
        Dataset kind.
    table_key
        Table key.
    owner
        Owner.
    freshness_sla
        Freshness SLA.
    retention_policy
        Retention policy.
    schema_version
        Schema version.
    validation_profile
        Validation profile.
    schema_id
        Schema ID.
    jsonl_filename
        JSONL filename.
    parquet_filename
        Parquet filename.
    stable_id
        Stable ID.
    specs_snapshot
        Specs snapshot path.
    output_dir
        Output directory.
    overwrite
        Whether to overwrite.
    dry_run
        Whether dry run.
    emit_bootstrap_snippet
        Whether to emit bootstrap.
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
    resolved_table_key = table_key or f"{'docs' if kind == 'view' else 'analytics'}.{name}"
    resolved_schema_id = schema_id or name
    resolved_stable_id = stable_id or name
    resolved_jsonl = jsonl_filename or (None if kind == "view" else f"{name}.jsonl")
    resolved_parquet = parquet_filename or (None if kind == "view" else f"{name}.parquet")
    existing_schema = Path("src/codeintel/config/schemas/export") / f"{resolved_schema_id}.json"

    if existing_schema.exists() and not overwrite:
        message = f"Schema already exists: {existing_schema}"
        raise ScaffoldConfigError(message, exit_code=1)

    if specs_snapshot.exists() and not overwrite:
        try:
            specs = json.loads(specs_snapshot.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            message = f"Failed to parse specs snapshot {specs_snapshot}: {exc}"
            raise ScaffoldConfigError(message, exit_code=2) from exc
        names = {str(spec.get("name")) for spec in specs}
        stable_ids = {str(spec.get("stable_id")) for spec in specs if "stable_id" in spec}
        if name in names:
            message = f"Dataset name already present in snapshot: {name}"
            raise ScaffoldConfigError(message, exit_code=1)
        if resolved_stable_id in stable_ids:
            message = f"Stable ID already present in snapshot: {resolved_stable_id}"
            raise ScaffoldConfigError(message, exit_code=1)

    if registry is not None:
        if name in registry.by_name:
            message = f"Dataset name already present in registry: {name}"
            raise ScaffoldConfigError(message, exit_code=1)
        if resolved_stable_id in {ds.stable_id for ds in registry.by_name.values() if ds.stable_id}:
            message = f"Stable ID already present in registry: {resolved_stable_id}"
            raise ScaffoldConfigError(message, exit_code=1)
        if resolved_table_key in registry.by_table_key:
            message = f"Table key already present in registry: {resolved_table_key}"
            raise ScaffoldConfigError(message, exit_code=1)

    return ScaffoldOptions(
        name=name,
        table_key=resolved_table_key,
        owner=owner,
        freshness_sla=freshness_sla,
        retention_policy=retention_policy,
        schema_version=schema_version,
        stable_id=resolved_stable_id,
        validation_profile=cast("Literal['strict', 'lenient']", validation_profile),
        jsonl_filename=resolved_jsonl,
        parquet_filename=resolved_parquet,
        schema_id=resolved_schema_id,
        output_dir=output_dir,
        is_view=kind == "view",
        overwrite=overwrite,
        dry_run=dry_run,
        emit_bootstrap_snippet=emit_bootstrap_snippet,
    )


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@datasets_ext_app.command("lint")
def datasets_lint(
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    schema_dir: SchemaDirOpt = Path("src/codeintel/config/schemas/export"),
    sample_rows: SampleRowsFlag = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Validate dataset contract health.

    Checks dataset contracts for schema consistency and integrity issues.

    Examples
    --------
    .. code-block:: bash

        # Validate contracts
        codeintel datasets lint

        # Validate with row sampling
        codeintel datasets lint --sample-rows
    """
    setup_logging(verbose)

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    issues = collect_contract_issues(gateway.con, schema_base_dir=schema_dir)

    if sample_rows:
        sys.stderr.write("Row sampling requested; run `codeintel datasets conformance` instead.\n")
        raise typer.Exit(code=2)

    if issues:
        for issue in issues:
            sys.stderr.write(f"{issue}\n")
        raise typer.Exit(code=1)

    typer.secho("Dataset contract validation passed.", fg=typer.colors.GREEN)


@datasets_ext_app.command("list")
def datasets_list(
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    docs_view: DocsViewFilterOpt = "include",
    read_only: ReadOnlyFilterOpt = "include",
    max_description: MaxDescriptionOpt = 80,
    verbose: VerboseOpt = 0,
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

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)

    rows: list[tuple[str, str, str, str, str]] = []
    for name, ds in sorted(registry.by_name.items()):
        caps = ds.capabilities()
        if not _caps_match(
            caps,
            docs_view_filter=docs_view,
            read_only_filter=read_only,
        ):
            continue
        rows.append(
            (
                name,
                ds.table_key,
                ds.family or "",
                _format_capabilities(caps),
                _truncate(ds.description or "", max_description),
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
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    verbose: VerboseOpt = 0,
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

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    specs = list_dataset_specs(load_dataset_registry(gateway.con))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    typer.secho(f"Wrote dataset specs to {output}", fg=typer.colors.GREEN)


@datasets_ext_app.command("diff")
def datasets_diff(
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    baseline: BaselineOpt = None,
    output: OutputOptional = None,
    against_ref: AgainstRefOpt = None,
    baseline_path: BaselinePathOpt = Path("build/dataset_specs.json"),
    verbose: VerboseOpt = 0,
) -> None:
    """Diff current dataset specs against a baseline.

    Compares current dataset specifications with a baseline snapshot
    to identify added, removed, or changed datasets.

    Examples
    --------
    .. code-block:: bash

        # Diff against file
        codeintel datasets diff --baseline build/baseline.json

        # Diff against git ref
        codeintel datasets diff --against-ref main
    """
    setup_logging(verbose)

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    current_specs = list_dataset_specs(load_dataset_registry(gateway.con))

    baseline_specs: list[dict[str, object]] = []
    if against_ref:
        baseline_specs = _load_specs_from_ref(
            repo_root=runtime.cfg.paths.repo_root,
            ref=against_ref,
            snapshot_path=baseline_path,
        )
    elif baseline is not None:
        if not baseline.exists():
            typer.secho(f"Baseline file not found: {baseline}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        baseline_specs = json.loads(baseline.read_text(encoding="utf-8"))
    else:
        typer.secho("Provide either --baseline or --against-ref", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    added, removed, changed = _diff_specs(current_specs, baseline_specs)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(current_specs, indent=2), encoding="utf-8")

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
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    schema_dir: SchemaDirOpt = Path("src/codeintel/config/schemas/export"),
    sample_rows: SampleRowsFlag = False,
    sample_size: SampleSizeOpt = 50,
    verbose: VerboseOpt = 0,
) -> None:
    """Run full dataset conformance checks.

    Includes contract validation and optional row sampling against
    JSON schemas.

    Examples
    --------
    .. code-block:: bash

        # Run conformance checks
        codeintel datasets conformance

        # Include row sampling
        codeintel datasets conformance --sample-rows --sample-size 100
    """
    setup_logging(verbose)

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)

    try:
        report = run_conformance(
            gateway.con,
            schema_base_dir=schema_dir,
            sample_rows=sample_rows,
            sample_size=sample_size,
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
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    output_dir: OutputDirOpt = Path("src/codeintel/config/schemas/export"),
    datasets: DatasetsFilterOpt = None,
    verbose: VerboseOpt = 0,
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

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)
    include = set(datasets) if datasets else None
    written = generate_export_schemas(
        registry,
        output_dir=output_dir,
        include_datasets=include,
    )
    if not written:
        sys.stdout.write("No schemas generated (no matching datasets with row bindings).\n")
        return
    typer.secho(f"Wrote {len(written)} schemas to {output_dir}", fg=typer.colors.GREEN)


@datasets_ext_app.command("catalog")
def datasets_catalog(
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    output_dir: OutputDirOpt = Path("build/catalog"),
    sample_rows_count: SampleRowsCountOpt = 3,
    sample_rows_strict: SampleRowsStrictFlag = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Generate Markdown/HTML dataset catalog.

    Creates documentation artifacts from the dataset registry.

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

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )

    if not runtime.paths.db_path.exists():
        if sample_rows_strict:
            typer.secho(
                f"Database not found at {runtime.paths.db_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        _warn(f"Database not found at {runtime.paths.db_path}; writing empty catalog.")
        output_dir.mkdir(parents=True, exist_ok=True)
        md = write_markdown_catalog(output_dir, [])
        html = write_html_catalog(output_dir, [])
        typer.secho(f"Wrote catalog: {md}, {html}", fg=typer.colors.GREEN)
        return

    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)

    try:
        entries = build_catalog(
            registry,
            con=gateway.con,
            sampling=SamplingConfig(
                sample_rows=sample_rows_count,
                sample_rows_strict=sample_rows_strict,
            ),
            warn=_warn,
        )
    except (DuckDBError, RuntimeError) as exc:
        typer.secho(f"Failed to generate catalog: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    md = write_markdown_catalog(output_dir, entries)
    html = write_html_catalog(output_dir, entries)
    typer.secho(f"Wrote catalog: {md}, {html}", fg=typer.colors.GREEN)


@datasets_ext_app.command("scaffold")
def datasets_scaffold(
    name: ScaffoldNameArg,
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    kind: ScaffoldKindOpt = "table",
    table_key: TableKeyOpt = None,
    owner: OwnerOpt = None,
    freshness_sla: FreshnessSlaOpt = None,
    retention_policy: RetentionPolicyOpt = None,
    schema_version: SchemaVersionOpt = "1",
    validation_profile: ValidationProfileOpt = "strict",
    schema_id: SchemaIdOpt = None,
    jsonl_filename: JsonlFilenameOpt = None,
    parquet_filename: ParquetFilenameOpt = None,
    stable_id: StableIdOpt = None,
    specs_snapshot: SpecsSnapshotOpt = Path("build/catalog/dataset_specs.json"),
    output_dir: OutputDirOpt = Path("build/dataset_scaffolds"),
    dry_run: DryRunFlag = False,
    overwrite: OverwriteFlag = False,
    emit_bootstrap_snippet: EmitBootstrapFlag = False,
    check_registry: CheckRegistryFlag = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Create a new dataset scaffold.

    Generates TypedDict, schema, bindings, and metadata files for a new dataset.

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
    if check_registry:
        runtime = build_runtime_or_exit(
            project_root=project_root,
            repo=repo,
            commit=commit,
            db_path=db_path,
            build_dir=build_dir,
            repo_root=repo_root,
        )
        gateway = open_gateway_from_config(runtime.cfg, read_only=True)
        registry = load_dataset_registry(gateway.con)

    try:
        opts = build_scaffold_options(
            name=name,
            kind=kind,
            table_key=table_key,
            owner=owner,
            freshness_sla=freshness_sla,
            retention_policy=retention_policy,
            schema_version=schema_version,
            validation_profile=validation_profile,
            schema_id=schema_id,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            stable_id=stable_id,
            specs_snapshot=specs_snapshot,
            output_dir=output_dir,
            overwrite=overwrite,
            dry_run=dry_run,
            emit_bootstrap_snippet=emit_bootstrap_snippet,
            registry=registry,
        )
    except ScaffoldConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
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
    schema_root: SchemaRootOpt = None,
    verbose: VerboseOpt = 0,
) -> None:
    """Validate exported JSONL/Parquet files against JSON Schemas.

    Validates one or more export files against a named JSON Schema definition.
    Supports both JSONL and Parquet file formats.

    Example
    -------
    .. code-block:: bash

        codeintel datasets validate-files --schema call_graph_edges exports/*.jsonl
        codeintel datasets validate-files --schema function_profile data.parquet
    """
    setup_logging(verbose)
    from codeintel.pipeline.export.validate_exports import (
        DEFAULT_SCHEMA_ROOT,
        validate_files,
    )

    root = schema_root if schema_root is not None else DEFAULT_SCHEMA_ROOT
    exit_code = validate_files(schema, files, schema_root=root)
    raise typer.Exit(code=exit_code)


__all__ = [
    "ScaffoldConfigError",
    "build_scaffold_options",
    "datasets_ext_app",
]
