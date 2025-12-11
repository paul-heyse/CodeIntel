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
from codeintel.cli.handlers.datasets import (
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_snapshot_handler,
)

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


__all__ = ["datasets_ext_app"]
