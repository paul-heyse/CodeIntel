"""Typer-free handlers for dataset management commands.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError`.

.. deprecated:: 2.0
    This module is deprecated. Use codeintel.cli.handlers.datasets instead.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from codeintel.cli.cli_errors import ProblemDetail, ValidationError

# Import consolidated setup_logging from handlers.base
from codeintel.cli.handlers.base import setup_logging as _setup_logging_impl
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)
from codeintel.config.models import CodeIntelConfig
from codeintel.export.validate_exports import (
    DEFAULT_SCHEMA_ROOT,
    validate_files,
)
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage import gateway as storage_gateway
from codeintel.storage.datasets import DatasetRegistry, list_dataset_specs, load_dataset_registry
from codeintel.storage.datasets.catalog import (
    SamplingConfig,
    build_catalog,
    write_html_catalog,
    write_markdown_catalog,
)
from codeintel.storage.datasets.scaffold import ScaffoldOptions, scaffold_dataset
from codeintel.storage.gateway import DuckDBError, StorageConfig, StorageGateway
from codeintel.storage.schema.json_schema import generate_export_schemas
from codeintel.storage.validation import collect_contract_issues
from codeintel.storage.validation.conformance import run_conformance

warnings.warn(
    "codeintel.cli.datasets_handlers is deprecated. Use codeintel.cli.handlers.datasets instead.",
    DeprecationWarning,
    stacklevel=2,
)

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class OutputFormat(Enum):
    """Output rendering format."""

    TEXT = "text"
    JSON = "json"


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


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeCliOptions:
    """CLI options for runtime resolution."""

    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None


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
# Logging Configuration
# -----------------------------------------------------------------------------

# Use consolidated setup_logging from handlers.base
setup_logging = _setup_logging_impl


# -----------------------------------------------------------------------------
# Runtime Utilities
# -----------------------------------------------------------------------------


def build_runtime_from_cli(options: RuntimeCliOptions) -> ProjectRuntime:
    """Build a ProjectRuntime from CLI options.

    Parameters
    ----------
    options
        CLI options containing project root.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.

    Raises
    ------
    ValidationError
        If the project cannot be resolved.
    """
    try:
        project_root = find_project_root(options.project_root)
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        message = f"Project not found: {exc}"
        raise ValidationError(message) from exc
    except Exception as exc:
        message = f"Failed to load project: {exc}"
        raise ValidationError(message) from exc


def open_gateway_from_config(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
    """Open a StorageGateway from CodeIntelConfig.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    read_only
        Whether to open read-only.

    Returns
    -------
    StorageGateway
        Opened gateway.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = (
        StorageConfig.for_readonly(cfg.paths.db_path)
        if read_only
        else StorageConfig.for_ingest(cfg.paths.db_path)
    )
    gateway_cfg = replace(
        base_cfg,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
    )
    return storage_gateway.open_gateway(gateway_cfg)


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


def _guard_existing_schema(
    existing_schema: Path,
    *,
    overwrite_allowed: bool,
    skip_overwrite: bool,
) -> ScaffoldConfigError | None:
    """Check if schema already exists and handle according to policy.

    Returns
    -------
    ScaffoldConfigError | None
        Error if conflict found, None otherwise.
    """
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
    """Check for name/stable_id conflicts in snapshot.

    Returns
    -------
    ScaffoldConfigError | None
        Error if conflict found, None otherwise.

    Raises
    ------
    ScaffoldConfigError
        If the specs snapshot cannot be parsed.
    """
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
    """Check for conflicts in live registry.

    Returns
    -------
    ScaffoldConfigError | None
        Error if conflict found, None otherwise.
    """
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
    """Build project selection from CLI values.

    Returns
    -------
    ProjectSelection
        Project selection options.
    """
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
    """Build build selection from CLI values.

    Returns
    -------
    BuildSelection
        Build selection options.
    """
    return BuildSelection(
        db_path=db_path,
        build_dir=build_dir,
    )


def _runtime_options(project: ProjectSelection, build: BuildSelection) -> RuntimeOptions:
    """Build runtime options from selections.

    Returns
    -------
    RuntimeOptions
        Combined runtime options.
    """
    return RuntimeOptions(project=project, build=build)


def _resolve_runtime(runtime: RuntimeOptions) -> ProjectRuntime:
    """Resolve runtime options to a ProjectRuntime.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.
    """
    cli_options = RuntimeCliOptions(
        project_root=runtime.project.project_root,
        repo=runtime.project.repo,
        commit=runtime.project.commit,
        db_path=runtime.build.db_path,
        build_dir=runtime.build.build_dir,
        repo_root=runtime.project.repo_root,
    )
    return build_runtime_from_cli(cli_options)


def _open_gateway(
    runtime: RuntimeOptions,
    *,
    read_only: bool = True,
) -> tuple[ProjectRuntime, StorageGateway]:
    """Open a gateway from runtime options.

    Returns
    -------
    tuple[ProjectRuntime, StorageGateway]
        Runtime and gateway.
    """
    runtime_state = _resolve_runtime(runtime)
    return runtime_state, open_gateway_from_config(runtime_state.cfg, read_only=read_only)


# -----------------------------------------------------------------------------
# Bundle Functions
# -----------------------------------------------------------------------------


def _runtime_from_kwargs(cli_kwargs: Mapping[str, object]) -> RuntimeOptions:
    """Extract runtime options from CLI kwargs.

    Returns
    -------
    RuntimeOptions
        Runtime options.
    """
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
    """Extract verbose level from CLI kwargs.

    Returns
    -------
    int
        Verbose level.
    """
    return int(cast("int | str | None", cli_kwargs.get("verbose", 0)) or 0)


def bundle_lint(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for lint command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
    runtime = _runtime_from_kwargs(cli_kwargs)
    lint = LintOptions(
        schema_dir=cast(
            "Path", cli_kwargs.get("schema_dir", Path("src/codeintel/config/schemas/export"))
        ),
        sampling=cast("SamplingMode", cli_kwargs.get("sample_rows", SamplingMode.DISABLED)),
    )
    return {"runtime": runtime, "lint": lint, "verbose": _verbose_from_kwargs(cli_kwargs)}


def bundle_list(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for list command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
    runtime = _runtime_from_kwargs(cli_kwargs)
    filters = ListFilters(
        docs_view=cast("str", cli_kwargs.get("docs_view", "include")),
        read_only=cast("str", cli_kwargs.get("read_only", "include")),
        max_description=int(cast("int | str | None", cli_kwargs.get("max_description", 80)) or 80),
    )
    return {"runtime": runtime, "filters": filters, "verbose": _verbose_from_kwargs(cli_kwargs)}


def bundle_snapshot(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for snapshot command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
    runtime = _runtime_from_kwargs(cli_kwargs)
    return {
        "runtime": runtime,
        "output": cast("Path", cli_kwargs["output"]),
        "verbose": _verbose_from_kwargs(cli_kwargs),
    }


def bundle_diff(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for diff command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
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


def bundle_conformance(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for conformance command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
    runtime = _runtime_from_kwargs(cli_kwargs)
    options = ConformanceOptions(
        schema_dir=cast(
            "Path", cli_kwargs.get("schema_dir", Path("src/codeintel/config/schemas/export"))
        ),
        sampling=cast("SamplingMode", cli_kwargs.get("sample_rows", SamplingMode.DISABLED)),
        sample_size=int(cast("int | str | None", cli_kwargs.get("sample_size", 50)) or 50),
    )
    return {"runtime": runtime, "options": options, "verbose": _verbose_from_kwargs(cli_kwargs)}


def bundle_generate_schemas(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for generate-schemas command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
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


def bundle_catalog(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for catalog command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
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


def bundle_scaffold(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for scaffold command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
    runtime = _runtime_from_kwargs(cli_kwargs)
    metadata = ScaffoldMetadataOptions(
        kind=cast("str", cli_kwargs.get("kind", "table")),
        table_key=cast("str | None", cli_kwargs.get("table_key")),
        owner=cast("str | None", cli_kwargs.get("owner")),
        freshness_sla=cast("str | None", cli_kwargs.get("freshness_sla")),
        retention_policy=cast("str | None", cli_kwargs.get("retention_policy")),
    )
    schema = ScaffoldSchemaOptions(
        schema_version=cast("str", cli_kwargs.get("schema_version", "1")),
        validation_profile=cast("str", cli_kwargs.get("validation_profile", "strict")),
        schema_id=cast("str | None", cli_kwargs.get("schema_id")),
    )
    files = ScaffoldFileOptions(
        jsonl_filename=cast("str | None", cli_kwargs.get("jsonl_filename")),
        parquet_filename=cast("str | None", cli_kwargs.get("parquet_filename")),
        stable_id=cast("str | None", cli_kwargs.get("stable_id")),
    )
    scaffold = DatasetScaffoldOptions(
        output_dir=cast("Path", cli_kwargs.get("output_dir", Path("build/dataset_scaffolds"))),
        overwrite_policy=cast(
            "OverwritePolicy", cli_kwargs.get("overwrite_policy", OverwritePolicy.ERROR)
        ),
    )
    io_opts = ScaffoldIOOptions(
        scaffold=scaffold,
        specs_snapshot=cast(
            "Path", cli_kwargs.get("specs_snapshot", Path("build/catalog/dataset_specs.json"))
        ),
    )
    behavior = ScaffoldBehaviorOptions(
        run_mode=(
            DryRunMode.DRY_RUN if bool(cli_kwargs.get("dry_run", False)) else DryRunMode.EXECUTE
        ),
        bootstrap=cast("BootstrapSnippet", cli_kwargs.get("bootstrap", BootstrapSnippet.SKIP)),
        registry_check=(
            RegistryCheck.ENABLED
            if bool(cli_kwargs.get("registry_check", False))
            else RegistryCheck.DISABLED
        ),
    )
    options = ScaffoldCliOptions(
        metadata=metadata,
        schema=schema,
        files=files,
        io=io_opts,
        behavior=behavior,
    )
    return {
        "name": cast("str", cli_kwargs["name"]),
        "runtime": runtime,
        "options": options,
        "verbose": _verbose_from_kwargs(cli_kwargs),
    }


def bundle_validate_files(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for validate-files command.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
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


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
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


@dataclass(frozen=True)
class DatasetConformanceResult:
    """Result from datasets conformance operation.

    Parameters
    ----------
    passed
        Whether conformance passed.
    issue_count
        Number of issues found.
    issues
        List of issue descriptions.
    """

    passed: bool
    issue_count: int
    issues: list[str]


@dataclass(frozen=True)
class DatasetGenerateSchemasResult:
    """Result from datasets generate-schemas operation.

    Parameters
    ----------
    written
        List of written schema paths.
    count
        Number of schemas generated.
    output_dir
        Output directory.
    """

    written: list[str]
    count: int
    output_dir: str


@dataclass(frozen=True)
class DatasetCatalogResult:
    """Result from datasets catalog operation.

    Parameters
    ----------
    md_path
        Path to generated Markdown catalog.
    html_path
        Path to generated HTML catalog.
    entries_count
        Number of catalog entries.
    """

    md_path: str
    html_path: str
    entries_count: int


@dataclass(frozen=True)
class DatasetScaffoldResult:
    """Result from datasets scaffold operation.

    Parameters
    ----------
    typed_dict
        Path to TypedDict file.
    row_binding
        Path to row binding snippet.
    json_schema
        Path to JSON schema.
    metadata
        Path to metadata file.
    bootstrap_snippet
        Path to bootstrap snippet.
    dry_run
        Whether this was a dry run.
    """

    typed_dict: str
    row_binding: str
    json_schema: str
    metadata: str
    bootstrap_snippet: str
    dry_run: bool


@dataclass(frozen=True)
class DatasetValidateFilesResult:
    """Result from datasets validate-files operation.

    Parameters
    ----------
    schema
        Schema name.
    files
        List of validated files.
    status
        Status (ok, failed, skipped, planned).
    exit_code
        Exit code from validation.
    """

    schema: str
    files: list[str]
    status: str
    exit_code: int


# -----------------------------------------------------------------------------
# ExecutionContext-based Handlers
# -----------------------------------------------------------------------------


def _build_runtime_from_ctx(ctx: ExecutionContext) -> ProjectRuntime:
    """Build ProjectRuntime from execution context.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.

    Raises
    ------
    RuntimeError
        If project cannot be resolved.
    """
    project_root_raw = ctx.params.get("project_root")
    project_root = Path(project_root_raw) if project_root_raw else None

    try:
        project_root_resolved = find_project_root(project_root)
        return build_project_runtime(project_root_resolved)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise RuntimeError(msg) from exc
    except Exception as exc:
        msg = f"Failed to load project: {exc}"
        raise RuntimeError(msg) from exc


def datasets_list_ctx(ctx: ExecutionContext) -> CliResult[DatasetsListResult]:
    """List datasets with capabilities and optional filters.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - docs_view: Filter for docs views (include, only, exclude).
        - read_only: Filter for read-only (include, only, exclude).
        - max_description: Maximum description length.

    Returns
    -------
    CliResult[DatasetsListResult]
        Result with list of datasets.
    """
    setup_logging(ctx.verbosity)

    runtime = _build_runtime_from_ctx(ctx)
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)

    docs_view_filter = ctx.get_str_param("docs_view", "include") or "include"
    read_only_filter = ctx.get_str_param("read_only", "include") or "include"
    max_description = ctx.get_int_param("max_description", 80)

    datasets: list[dict[str, Any]] = []
    for name, ds in sorted(registry.by_name.items()):
        caps = ds.capabilities()
        if not _caps_match(
            caps,
            docs_view_filter=docs_view_filter,
            read_only_filter=read_only_filter,
        ):
            continue
        datasets.append(
            {
                "name": name,
                "table_key": ds.table_key,
                "family": ds.family or "",
                "capabilities": _format_capabilities(caps),
                "description": _truncate(ds.description or "", max_description),
            }
        )

    return CliResult.ok(
        DatasetsListResult(
            datasets=datasets,
            count=len(datasets),
        )
    )


def datasets_lint_ctx(ctx: ExecutionContext) -> CliResult[DatasetLintResult]:
    """Validate dataset contract health.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - schema_dir: Schema directory.
        - sampling: Sampling mode.

    Returns
    -------
    CliResult[DatasetLintResult]
        Result with validation status.
    """
    setup_logging(ctx.verbosity)

    runtime = _build_runtime_from_ctx(ctx)
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)

    schema_dir_raw = ctx.params.get("schema_dir")
    schema_dir = Path(schema_dir_raw) if schema_dir_raw else None
    issues = collect_contract_issues(gateway.con, schema_base_dir=schema_dir)

    sampling_raw = ctx.get_str_param("sampling", "disabled")
    if sampling_raw == "enabled":
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Invalid Option",
                detail="Row sampling not supported in lint; use conformance command",
                status=400,
            )
        )

    if issues:
        return CliResult.ok(
            DatasetLintResult(
                passed=False,
                issue_count=len(issues),
                issues=[str(issue) for issue in issues],
            )
        )

    return CliResult.ok(
        DatasetLintResult(
            passed=True,
            issue_count=0,
            issues=[],
        )
    )


def datasets_snapshot_ctx(ctx: ExecutionContext) -> CliResult[DatasetSnapshotResult]:
    """Write current dataset specs to a JSON snapshot file.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - output: Output path.

    Returns
    -------
    CliResult[DatasetSnapshotResult]
        Result with snapshot information.
    """
    setup_logging(ctx.verbosity)

    runtime = _build_runtime_from_ctx(ctx)
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    specs = list_dataset_specs(load_dataset_registry(gateway.con))

    output_raw = ctx.params.get("output")
    if output_raw is None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Missing Parameter",
                detail="Output path is required",
                status=400,
            )
        )
    output = Path(output_raw)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specs, indent=2), encoding="utf-8")

    return CliResult.ok(
        DatasetSnapshotResult(
            output_path=str(output),
            datasets_count=len(specs),
        )
    )


def _load_diff_baseline(
    ctx: ExecutionContext,
    runtime: ProjectRuntime,
) -> list[dict[str, object]] | None:
    """Load baseline specs for diff operation.

    Parameters
    ----------
    ctx
        Execution context.
    runtime
        Project runtime.

    Returns
    -------
    list[dict[str, object]] | None
        Baseline specs or None if error.
    """
    against_ref = ctx.get_str_param("against_ref")
    baseline_raw = ctx.params.get("baseline")
    baseline_path_raw = ctx.params.get("baseline_path")

    if against_ref:
        baseline_path = (
            Path(baseline_path_raw)
            if baseline_path_raw
            else Path("build/catalog/dataset_specs.json")
        )
        return _load_specs_from_ref(
            repo_root=runtime.cfg.paths.repo_root,
            ref=against_ref,
            snapshot_path=baseline_path,
        )

    if baseline_raw is not None:
        baseline = Path(baseline_raw)
        if not baseline.exists():
            return None
        return cast(
            "list[dict[str, object]]",
            json.loads(baseline.read_text(encoding="utf-8")),
        )

    return None


def datasets_diff_ctx(ctx: ExecutionContext) -> CliResult[DatasetDiffResult]:
    """Diff current dataset specs against a baseline.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - baseline: Baseline file path.
        - against_ref: Git ref to compare against.
        - baseline_path: Path within ref for baseline.
        - output: Optional output path for current specs.

    Returns
    -------
    CliResult[DatasetDiffResult]
        Result with diff information.
    """
    setup_logging(ctx.verbosity)

    runtime = _build_runtime_from_ctx(ctx)
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    current_specs = list_dataset_specs(load_dataset_registry(gateway.con))

    baseline_specs = _load_diff_baseline(ctx, runtime)
    if baseline_specs is None:
        against_ref = ctx.get_str_param("against_ref")
        baseline_raw = ctx.params.get("baseline")
        if against_ref is None and baseline_raw is None:
            return CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:cli:validation-error",
                    title="Missing Parameter",
                    detail="Provide either --baseline or --against-ref",
                    status=400,
                )
            )
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:not-found",
                title="File Not Found",
                detail=f"Baseline file not found: {baseline_raw}",
                status=404,
            )
        )

    added, removed, changed = _diff_specs(current_specs, baseline_specs)

    output_raw = ctx.params.get("output")
    if output_raw is not None:
        output = Path(output_raw)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(current_specs, indent=2), encoding="utf-8")

    return CliResult.ok(
        DatasetDiffResult(
            added=list(added),
            removed=list(removed),
            changed=list(changed),
            has_differences=bool(added or removed or changed),
        )
    )


def datasets_conformance_ctx(
    ctx: ExecutionContext,
) -> CliResult[DatasetConformanceResult]:
    """Run full dataset conformance checks.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - schema_dir: Schema directory.
        - sampling: Sampling mode.
        - sample_size: Sample size.

    Returns
    -------
    CliResult[DatasetConformanceResult]
        Result with conformance status.
    """
    setup_logging(ctx.verbosity)

    runtime = _build_runtime_from_ctx(ctx)
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)

    schema_dir_raw = ctx.params.get("schema_dir")
    schema_dir = Path(schema_dir_raw) if schema_dir_raw else DEFAULT_SCHEMA_ROOT
    sampling_raw = ctx.get_str_param("sampling", "disabled")
    sample_size = ctx.get_int_param("sample_size", 100)

    try:
        report = run_conformance(
            gateway.con,
            schema_base_dir=schema_dir,
            sample_rows=sampling_raw == "enabled",
            sample_size=sample_size,
        )
    except (DuckDBError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:operation-error",
                title="Conformance Error",
                detail=f"Conformance run failed: {exc}",
                status=500,
            )
        )

    if not report.ok:
        issues = [f"[{issue.dataset or 'global'}] {issue.message}" for issue in report.issues]
        return CliResult.ok(
            DatasetConformanceResult(
                passed=False,
                issue_count=len(issues),
                issues=issues,
            )
        )

    return CliResult.ok(
        DatasetConformanceResult(
            passed=True,
            issue_count=0,
            issues=[],
        )
    )


def datasets_generate_schemas_ctx(
    ctx: ExecutionContext,
) -> CliResult[DatasetGenerateSchemasResult]:
    """Generate export JSON Schemas from TypedDict row models.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - datasets: Optional list of datasets to include.
        - output_dir: Output directory.

    Returns
    -------
    CliResult[DatasetGenerateSchemasResult]
        Result with generation information.
    """
    setup_logging(ctx.verbosity)

    runtime = _build_runtime_from_ctx(ctx)
    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)

    datasets_raw = ctx.params.get("datasets")
    include = set(datasets_raw) if datasets_raw else None
    output_dir_raw = ctx.params.get("output_dir")
    output_dir = Path(output_dir_raw) if output_dir_raw else Path("schema/export")

    written = generate_export_schemas(
        registry,
        output_dir=output_dir,
        include_datasets=include,
    )

    return CliResult.ok(
        DatasetGenerateSchemasResult(
            written=[str(path) for path in written],
            count=len(written),
            output_dir=str(output_dir),
        )
    )


def datasets_catalog_ctx(ctx: ExecutionContext) -> CliResult[DatasetCatalogResult]:
    """Generate Markdown/HTML dataset catalog.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - output_dir: Output directory.
        - sample_rows_count: Number of sample rows.
        - sample_rows_strict: Strictness policy.

    Returns
    -------
    CliResult[DatasetCatalogResult]
        Result with catalog information.
    """
    setup_logging(ctx.verbosity)

    runtime = _build_runtime_from_ctx(ctx)
    output_dir_raw = ctx.params.get("output_dir")
    output_dir = Path(output_dir_raw) if output_dir_raw else Path("docs/datasets")
    sample_rows_count = ctx.get_int_param("sample_rows_count", 5)
    strict_raw = ctx.get_str_param("sample_rows_strict", "lenient")
    strict = strict_raw == "strict"

    if not runtime.paths.db_path.exists():
        if strict:
            return CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:cli:not-found",
                    title="Database Not Found",
                    detail=f"Database not found at {runtime.paths.db_path}",
                    status=404,
                )
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        md = write_markdown_catalog(output_dir, [])
        html = write_html_catalog(output_dir, [])
        return CliResult.ok(
            DatasetCatalogResult(
                md_path=str(md),
                html_path=str(html),
                entries_count=0,
            )
        )

    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)

    try:
        entries = build_catalog(
            registry,
            con=gateway.con,
            sampling=SamplingConfig(
                sample_rows=sample_rows_count,
                sample_rows_strict=strict,
            ),
        )
    except (DuckDBError, RuntimeError) as exc:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:operation-error",
                title="Catalog Generation Error",
                detail=f"Failed to generate catalog: {exc}",
                status=500,
            )
        )

    md = write_markdown_catalog(output_dir, entries)
    html = write_html_catalog(output_dir, entries)

    return CliResult.ok(
        DatasetCatalogResult(
            md_path=str(md),
            html_path=str(html),
            entries_count=len(entries),
        )
    )


def datasets_scaffold_ctx(ctx: ExecutionContext) -> CliResult[DatasetScaffoldResult]:
    """Create a new dataset scaffold.

    Parameters
    ----------
    ctx
        Execution context with params:
        - name: Dataset name.
        - project_root: Optional project root override.
        - registry_check: Whether to check registry.
        - dry_run: Whether this is a dry run.
        - Various scaffold options.

    Returns
    -------
    CliResult[DatasetScaffoldResult]
        Result with scaffold information.
    """
    setup_logging(ctx.verbosity)

    name = ctx.params.get("name")
    if name is None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Missing Parameter",
                detail="Dataset name is required",
                status=400,
            )
        )

    registry: DatasetRegistry | None = None
    registry_check_raw = ctx.get_str_param("registry_check", "enabled")
    if registry_check_raw == "enabled":
        runtime = _build_runtime_from_ctx(ctx)
        gateway = open_gateway_from_config(runtime.cfg, read_only=True)
        registry = load_dataset_registry(gateway.con)

    # Build scaffold CLI options from ctx params
    options = _build_scaffold_cli_options(ctx)

    try:
        opts = build_scaffold_options(
            name=name,
            options=options,
            registry=registry,
        )
    except ScaffoldConfigError as exc:
        if exc.exit_code == 0:
            return CliResult.ok(
                DatasetScaffoldResult(
                    typed_dict="",
                    row_binding="",
                    json_schema="",
                    metadata="",
                    bootstrap_snippet="",
                    dry_run=ctx.dry_run,
                )
            )
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Scaffold Error",
                detail=str(exc),
                status=400,
            )
        )

    result = scaffold_dataset(opts)

    return CliResult.ok(
        DatasetScaffoldResult(
            typed_dict=str(result.typed_dict),
            row_binding=str(result.row_binding),
            json_schema=str(result.json_schema),
            metadata=str(result.metadata),
            bootstrap_snippet=str(result.bootstrap_snippet),
            dry_run=opts.dry_run,
        )
    )


def _build_scaffold_cli_options(ctx: ExecutionContext) -> ScaffoldCliOptions:
    """Build ScaffoldCliOptions from execution context.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    ScaffoldCliOptions
        Scaffold CLI options.
    """
    output_dir_raw = ctx.params.get("output_dir")
    specs_snapshot_raw = ctx.params.get("specs_snapshot")
    overwrite_raw = ctx.get_str_param("overwrite_policy", "error")
    bootstrap_raw = ctx.get_str_param("bootstrap", "skip")
    registry_check_raw = ctx.get_str_param("registry_check", "disabled")

    return ScaffoldCliOptions(
        metadata=ScaffoldMetadataOptions(
            kind=ctx.get_str_param("kind", "table") or "table",
            table_key=ctx.get_str_param("table_key"),
            owner=ctx.get_str_param("owner"),
            freshness_sla=ctx.get_str_param("freshness_sla"),
            retention_policy=ctx.get_str_param("retention_policy"),
        ),
        schema=ScaffoldSchemaOptions(
            schema_version=ctx.get_str_param("schema_version", "1") or "1",
            validation_profile=ctx.get_str_param("validation_profile", "strict") or "strict",
            schema_id=ctx.get_str_param("schema_id"),
        ),
        files=ScaffoldFileOptions(
            jsonl_filename=ctx.get_str_param("jsonl_filename"),
            parquet_filename=ctx.get_str_param("parquet_filename"),
            stable_id=ctx.get_str_param("stable_id"),
        ),
        io=ScaffoldIOOptions(
            scaffold=DatasetScaffoldOptions(
                output_dir=(
                    Path(output_dir_raw) if output_dir_raw else Path("build/dataset_scaffolds")
                ),
                overwrite_policy=(
                    OverwritePolicy(overwrite_raw) if overwrite_raw else OverwritePolicy.ERROR
                ),
            ),
            specs_snapshot=(
                Path(specs_snapshot_raw)
                if specs_snapshot_raw
                else Path("build/catalog/dataset_specs.json")
            ),
        ),
        behavior=ScaffoldBehaviorOptions(
            run_mode=DryRunMode.DRY_RUN if ctx.dry_run else DryRunMode.EXECUTE,
            bootstrap=(BootstrapSnippet(bootstrap_raw) if bootstrap_raw else BootstrapSnippet.SKIP),
            registry_check=(
                RegistryCheck.ENABLED if registry_check_raw == "enabled" else RegistryCheck.DISABLED
            ),
        ),
    )


def datasets_validate_files_ctx(
    ctx: ExecutionContext,
) -> CliResult[DatasetValidateFilesResult]:
    """Validate exported JSONL/Parquet files against JSON Schemas.

    Parameters
    ----------
    ctx
        Execution context with params:
        - schema: Schema name.
        - files: Files to validate.
        - schema_root: Schema root directory.
        - validation: Validation mode.
        - dry_run: Whether this is a dry run.

    Returns
    -------
    CliResult[DatasetValidateFilesResult]
        Result with validation information.
    """
    setup_logging(ctx.verbosity)

    schema = ctx.params.get("schema")
    if schema is None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Missing Parameter",
                detail="Schema name is required",
                status=400,
            )
        )

    files_raw = ctx.params.get("files", [])
    files = [Path(f) for f in files_raw]

    schema_root_raw = ctx.params.get("schema_root")
    root = Path(schema_root_raw) if schema_root_raw else DEFAULT_SCHEMA_ROOT

    validation_raw = ctx.get_str_param("validation", "required")

    if ctx.dry_run:
        return CliResult.ok(
            DatasetValidateFilesResult(
                schema=schema,
                files=[str(f) for f in files],
                status="planned",
                exit_code=0,
            )
        )

    if validation_raw == "skip":
        return CliResult.ok(
            DatasetValidateFilesResult(
                schema=schema,
                files=[str(f) for f in files],
                status="skipped",
                exit_code=0,
            )
        )

    exit_code = validate_files(schema, files, schema_root=root)

    return CliResult.ok(
        DatasetValidateFilesResult(
            schema=schema,
            files=[str(f) for f in files],
            status="ok" if exit_code == 0 else "failed",
            exit_code=exit_code,
        )
    )


__all__ = [
    "BootstrapSnippet",
    "BuildSelection",
    "CatalogOptions",
    "ConformanceOptions",
    "DatasetCatalogResult",
    "DatasetConformanceResult",
    "DatasetDiffResult",
    "DatasetExportOptions",
    "DatasetGenerateSchemasResult",
    "DatasetLintResult",
    "DatasetScaffoldOptions",
    "DatasetScaffoldResult",
    "DatasetSnapshotResult",
    "DatasetValidateFilesResult",
    "DatasetsListResult",
    "DiffOptions",
    "DryRunMode",
    "ExportOutputOptions",
    "ExportSelectionOptions",
    "ExportValidationMode",
    "ExportValidationOptions",
    "GenerateSchemasOptions",
    "LintOptions",
    "ListFilters",
    "MacroRequirement",
    "OutputFormat",
    "OverwritePolicy",
    "ProjectSelection",
    "RegistryCheck",
    "RuntimeCliOptions",
    "RuntimeOptions",
    "SamplingMode",
    "SamplingStrictness",
    "ScaffoldBehaviorOptions",
    "ScaffoldCliOptions",
    "ScaffoldConfigError",
    "ScaffoldFileOptions",
    "ScaffoldIOOptions",
    "ScaffoldMetadataOptions",
    "ScaffoldSchemaOptions",
    "build_scaffold_options",
    "bundle_catalog",
    "bundle_conformance",
    "bundle_diff",
    "bundle_generate_schemas",
    "bundle_lint",
    "bundle_list",
    "bundle_scaffold",
    "bundle_snapshot",
    "bundle_validate_files",
    "datasets_catalog_ctx",
    "datasets_conformance_ctx",
    "datasets_diff_ctx",
    "datasets_generate_schemas_ctx",
    "datasets_lint_ctx",
    "datasets_list_ctx",
    "datasets_scaffold_ctx",
    "datasets_snapshot_ctx",
    "datasets_validate_files_ctx",
    "setup_logging",
]
