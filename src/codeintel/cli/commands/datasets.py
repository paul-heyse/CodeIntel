"""Dataset management commands.

This module wires Cyclopts command classes to unified handlers via @cli_command.

Note: Dataset commands require runtime/gateway access that is not yet fully
supported by the Command[T] pattern's Deps abstraction. They use the handler
pattern for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import App

from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    iter_contracts,
)
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.errors.builder import ProblemBuilder
from codeintel.cli.handlers.datasets import (
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_snapshot_handler,
)
from codeintel.cli.options.registry import (
    DATASETS_DIFF_AGAINST_REF,
    DATASETS_DIFF_BASELINE,
    DATASETS_DIFF_BASELINE_PATH,
    DATASETS_DIFF_OUTPUT,
    DATASETS_DOCS_VIEW,
    DATASETS_MAX_DESCRIPTION,
    DATASETS_READ_ONLY,
    DATASETS_SAMPLING,
    DATASETS_SCAFFOLD_DRY_RUN,
    DATASETS_SCAFFOLD_NAME,
    DATASETS_SCAFFOLD_REGISTRY_CHECK,
    DATASETS_SCHEMA_DIR,
    DATASETS_SNAPSHOT_OUTPUT,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param
from codeintel.core.errors.taxonomy import OperationErrorCode

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

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


_DATASETS_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)
_SCAFFOLD_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

DATASETS_LINT_PATH: CommandPath = ("datasets", "lint")
DATASETS_LIST_PATH: CommandPath = ("datasets", "list")
DATASETS_SNAPSHOT_PATH: CommandPath = ("datasets", "snapshot")
DATASETS_DIFF_PATH: CommandPath = ("datasets", "diff")
DATASETS_SCAFFOLD_PATH: CommandPath = ("datasets", "scaffold")

_DATASETS_LINT_FLAGS_FIELD = shared_flags_field(DATASETS_LINT_PATH)
_DATASETS_LIST_FLAGS_FIELD = shared_flags_field(DATASETS_LIST_PATH)
_DATASETS_SNAPSHOT_FLAGS_FIELD = shared_flags_field(DATASETS_SNAPSHOT_PATH)
_DATASETS_DIFF_FLAGS_FIELD = shared_flags_field(DATASETS_DIFF_PATH)
_DATASETS_SCAFFOLD_FLAGS_FIELD = shared_flags_field(DATASETS_SCAFFOLD_PATH)


@cli_command("datasets.lint", handler=datasets_lint_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="lint")
@dataclass
class LintCommand:
    """Validate dataset contract health."""

    schema_dir: Annotated[
        Path,
        option_param(DATASETS_SCHEMA_DIR, command_path=DATASETS_LINT_PATH),
    ] = Path("src/codeintel/config/schemas/export")
    sampling: Annotated[
        str,
        option_param(DATASETS_SAMPLING, command_path=DATASETS_LINT_PATH),
    ] = "disabled"
    flags: SharedFlagsProtocol = _DATASETS_LINT_FLAGS_FIELD


@cli_command("datasets.list", handler=datasets_list_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="list")
@dataclass
class ListDatasetsCommand:
    """List datasets with capabilities and optional filters."""

    docs_view: Annotated[
        DocsFilterMode,
        option_param(DATASETS_DOCS_VIEW, command_path=DATASETS_LIST_PATH),
    ] = "include"
    read_only: Annotated[
        ReadOnlyFilterMode,
        option_param(DATASETS_READ_ONLY, command_path=DATASETS_LIST_PATH),
    ] = "include"
    max_description: Annotated[
        int,
        option_param(DATASETS_MAX_DESCRIPTION, command_path=DATASETS_LIST_PATH),
    ] = 80
    flags: SharedFlagsProtocol = _DATASETS_LIST_FLAGS_FIELD


@cli_command("datasets.snapshot", handler=datasets_snapshot_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="snapshot")
@dataclass
class SnapshotCommand:
    """Write current dataset specs to a JSON snapshot file."""

    output: Annotated[
        Path,
        option_param(DATASETS_SNAPSHOT_OUTPUT, command_path=DATASETS_SNAPSHOT_PATH),
    ] = Path("build/dataset_specs.json")
    flags: SharedFlagsProtocol = _DATASETS_SNAPSHOT_FLAGS_FIELD


@cli_command("datasets.diff", handler=datasets_diff_handler, config=_DATASETS_CONFIG)
@datasets_ext_app.command(name="diff")
@dataclass
class DiffCommand:
    """Diff current dataset specs against a baseline."""

    baseline: Annotated[
        Path | None,
        option_param(DATASETS_DIFF_BASELINE, command_path=DATASETS_DIFF_PATH),
    ] = None
    output: Annotated[
        Path | None,
        option_param(DATASETS_DIFF_OUTPUT, command_path=DATASETS_DIFF_PATH),
    ] = None
    against_ref: Annotated[
        str | None,
        option_param(DATASETS_DIFF_AGAINST_REF, command_path=DATASETS_DIFF_PATH),
    ] = None
    baseline_path: Annotated[
        Path,
        option_param(DATASETS_DIFF_BASELINE_PATH, command_path=DATASETS_DIFF_PATH),
    ] = Path("build/dataset_specs.json")
    flags: SharedFlagsProtocol = _DATASETS_DIFF_FLAGS_FIELD


@cli_command("datasets.scaffold", config=_SCAFFOLD_CONFIG)
@datasets_ext_app.command(name="scaffold")
@dataclass(frozen=True)
class ScaffoldDatasetCommand(Command[dict[str, object]]):
    """Scaffold a new dataset definition."""

    name: Annotated[
        str,
        option_param(DATASETS_SCAFFOLD_NAME, command_path=DATASETS_SCAFFOLD_PATH),
    ]
    registry_check: Annotated[
        Literal["enabled", "disabled"],
        option_param(DATASETS_SCAFFOLD_REGISTRY_CHECK, command_path=DATASETS_SCAFFOLD_PATH),
    ] = "enabled"
    dry_run: Annotated[
        bool,
        option_param(DATASETS_SCAFFOLD_DRY_RUN, command_path=DATASETS_SCAFFOLD_PATH),
    ] = False
    flags: SharedFlagsProtocol = _DATASETS_SCAFFOLD_FLAGS_FIELD

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
        settings = ContractResolutionSettings(mode=ContractResolutionMode.FULL)
        contracts_by_table_key = {
            contract.table_key: contract for contract in iter_contracts(settings=settings)
        }
        known_names = set(contracts_by_table_key)
        known_names.update(key.split(".", 1)[-1] for key in contracts_by_table_key)
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
