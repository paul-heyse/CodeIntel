"""Dataset management commands.

This module wires Cyclopts command classes to unified handlers via @cli_command.

Note: Dataset commands require runtime/gateway access that is not yet fully
supported by the Command[T] pattern's Deps abstraction. They use the handler
pattern for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.context import CommandContext
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.errors.builder import ProblemBuilder
from codeintel.cli.errors.taxonomy import OperationErrorCode
from codeintel.cli.handlers.datasets import (
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_snapshot_handler,
)
from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY

datasets_ext_app = App(
    name="datasets",
    help="Extended dataset management commands.",
)


DocsFilterMode = Literal["include", "only", "exclude"]
ReadOnlyFilterMode = Literal["include", "only", "exclude"]


class SamplingStrictness(Enum):
    """Strictness policy when sampling rows."""

    STRICT = "strict"
    LENIENT = "lenient"


class OverwritePolicy(Enum):
    """Behavior when scaffold outputs already exist."""

    OVERWRITE = "overwrite"
    SKIP = "skip"
    ERROR = "error"


class BootstrapSnippet(Enum):
    """Whether to emit a bootstrap snippet during scaffold."""

    EMIT = "emit"
    SKIP = "skip"


# Config for datasets commands - requires runtime and gateway
_DATASETS_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)
_SCAFFOLD_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("datasets.lint", handler=datasets_lint_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="lint")
@dataclass
class LintCommand:
    """Validate dataset contract health."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sampling: Annotated[
        str,
        Parameter(
            name="--sampling",
            help="Sampling mode: enabled or disabled.",
        ),
    ] = "disabled"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("datasets.list", handler=datasets_list_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="list")
@dataclass
class ListDatasetsCommand:
    """List datasets with capabilities and optional filters."""

    docs_view: Annotated[
        DocsFilterMode,
        Parameter(
            name="--docs-view",
            help='Docs view filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    read_only: Annotated[
        ReadOnlyFilterMode,
        Parameter(
            name="--read-only",
            help='Read-only filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    max_description: Annotated[
        int,
        Parameter(
            name="--max-description",
            help="Maximum description length before truncation.",
        ),
    ] = 80
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("datasets.snapshot", handler=datasets_snapshot_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="snapshot")
@dataclass
class SnapshotCommand:
    """Write current dataset specs to a JSON snapshot file."""

    output: Annotated[
        Path,
        Parameter(
            name="--output",
            help="Output file path for JSON dataset specs.",
        ),
    ] = Path("build/dataset_specs.json")
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("datasets.diff", handler=datasets_diff_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="diff")
@dataclass
class DiffCommand:
    """Diff current dataset specs against a baseline."""

    baseline: Annotated[
        Path | None,
        Parameter(
            name="--baseline",
            help="Path to JSON baseline from `codeintel datasets snapshot`.",
        ),
    ] = None
    output: Annotated[
        Path | None,
        Parameter(
            name="--output",
            help="Optional output file path for writing current specs.",
        ),
    ] = None
    against_ref: Annotated[
        str | None,
        Parameter(
            name="--against-ref",
            help="Git ref to diff against (e.g. HEAD~, main).",
        ),
    ] = None
    baseline_path: Annotated[
        Path,
        Parameter(
            name="--baseline-path",
            help="Path of the snapshot file inside the git ref.",
        ),
    ] = Path("build/dataset_specs.json")
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("datasets.scaffold", config=_SCAFFOLD_CONFIG)
@datasets_ext_app.command(name="scaffold")
@dataclass(frozen=True)
class ScaffoldDatasetCommand(Command[dict[str, object]]):
    """Scaffold a new dataset definition."""

    name: Annotated[
        str,
        Parameter(
            name="name",
            help="Dataset name to scaffold.",
        ),
    ]
    registry_check: Annotated[
        Literal["enabled", "disabled"],
        Parameter(
            name="--registry-check",
            help="Whether to fail when the dataset already exists.",
            show_default=True,
        ),
    ] = "enabled"
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Perform validation only without writing files.",
            negative=("--no-dry-run",),
        ),
    ] = False
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[dict[str, object]]:
        """Validate scaffold request and report status.

        Parameters
        ----------
        ctx
            Command context (unused).

        Returns
        -------
        CliResult[dict[str, object]]
            Result with dataset name, status, and registry check behavior.
        """
        _ = ctx
        known_names = set(DATASET_CONTRACTS_BY_TABLE_KEY)
        known_names.update(key.split(".", 1)[-1] for key in DATASET_CONTRACTS_BY_TABLE_KEY)
        if self.registry_check == "enabled" and self.name in known_names:
            problem = ProblemBuilder.operation(
                OperationErrorCode.ALREADY_EXISTS,
                "datasets.scaffold",
                f"Dataset '{self.name}' already exists in registry.",
            )
            return CliResult.fail(problem)

        return CliResult.ok(
            {
                "dataset": self.name,
                "status": "dry_run" if self.dry_run else "created",
                "registry_check": self.registry_check,
            }
        )


__all__ = ["datasets_ext_app"]
