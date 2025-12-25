"""Registry inspection commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import App

from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.results import result_type
from codeintel.cli.errors.results import fail_invalid_value, fail_not_found
from codeintel.cli.options.registry import (
    REGISTRY_INVENTORY_PATH,
    REGISTRY_OUTPUTS_DOMAIN,
    REGISTRY_OUTPUTS_MATERIALIZATION,
    REGISTRY_OUTPUTS_PILOT_ONLY,
    REGISTRY_OUTPUTS_TABLE_KEY,
    REGISTRY_OUTPUTS_TARGETS,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param
from codeintel.core.registry.service import DagOutputInventory, DagOutputSpec, RegistryService

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)

registry_app = App(
    name="registry",
    help="Registry inspection commands.",
)

REGISTRY_OUTPUTS_PATH: CommandPath = ("registry", "outputs")
REGISTRY_VALIDATE_PATH: CommandPath = ("registry", "validate")

_REGISTRY_OUTPUTS_FLAGS_FIELD = shared_flags_field(REGISTRY_OUTPUTS_PATH)
_REGISTRY_VALIDATE_FLAGS_FIELD = shared_flags_field(REGISTRY_VALIDATE_PATH)

RegistryDomain = Literal["analytics", "graphs", "ingestion", "export"]
MaterializationKind = Literal["table", "artifact", "mixed"]


@dataclass(frozen=True)
class _RegistryOutputFilters:
    domain: RegistryDomain | None
    materialization: MaterializationKind | None
    targets: set[str] | None
    table_key: str | None
    pilot_only: bool


@result_type
@dataclass(frozen=True)
class DagOutputInfo:
    """Summary information about a DAG output target."""

    target: str
    domain: str
    anchor: str
    materialization: str
    table_keys: list[str]
    contracts: list[str]
    upstream_targets: list[str]
    downstream_consumers: list[str]
    pilot: bool
    notes: str | None = None


@result_type
@dataclass(frozen=True)
class DagOutputListResult:
    """Result from listing DAG output inventory."""

    outputs: list[DagOutputInfo]
    count: int
    inventory_path: str


@result_type
@dataclass(frozen=True)
class DagOutputValidateResult:
    """Result from validating DAG output inventory."""

    inventory_path: str
    output_count: int
    pilot_targets: list[str]
    version: int
    generated_at: str | None = None


def _inventory_path_or_default(path: Path | None) -> Path:
    return path or RegistryService.default_dag_output_inventory_path()


def _load_inventory(path: Path) -> DagOutputInventory:
    if not path.exists():
        raise FileNotFoundError(path)
    return RegistryService.load_dag_output_inventory(path=path)


def _spec_matches_filters(
    spec: DagOutputSpec,
    *,
    filters: _RegistryOutputFilters,
) -> bool:
    if filters.domain is not None and spec.domain != filters.domain:
        return False
    if filters.materialization is not None and spec.materialization != filters.materialization:
        return False
    if filters.targets is not None and spec.target not in filters.targets:
        return False
    if filters.table_key is not None and filters.table_key not in spec.table_keys:
        return False
    return not (filters.pilot_only and not spec.pilot)


@cli_command("registry.outputs", require_storage=False)
@registry_app.command(name="outputs")
@dataclass(frozen=True)
class RegistryOutputs(Command[DagOutputListResult]):
    """List DAG output inventory entries."""

    __operation_id__ = "registry.outputs"

    inventory_path: Annotated[
        Path | None,
        option_param(REGISTRY_INVENTORY_PATH, command_path=REGISTRY_OUTPUTS_PATH),
    ] = None
    domain: Annotated[
        RegistryDomain | None,
        option_param(REGISTRY_OUTPUTS_DOMAIN, command_path=REGISTRY_OUTPUTS_PATH),
    ] = None
    materialization: Annotated[
        MaterializationKind | None,
        option_param(REGISTRY_OUTPUTS_MATERIALIZATION, command_path=REGISTRY_OUTPUTS_PATH),
    ] = None
    targets: Annotated[
        list[str] | None,
        option_param(REGISTRY_OUTPUTS_TARGETS, command_path=REGISTRY_OUTPUTS_PATH),
    ] = None
    table_key: Annotated[
        str | None,
        option_param(REGISTRY_OUTPUTS_TABLE_KEY, command_path=REGISTRY_OUTPUTS_PATH),
    ] = None
    pilot_only: Annotated[
        bool,
        option_param(REGISTRY_OUTPUTS_PILOT_ONLY, command_path=REGISTRY_OUTPUTS_PATH),
    ] = False
    flags: SharedFlagsProtocol = _REGISTRY_OUTPUTS_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[DagOutputListResult]:
        """Execute registry output listing.

        Parameters
        ----------
        ctx
            CLI execution context.

        Returns
        -------
        CliResult[DagOutputListResult]
            Registry output listing result.
        """
        _ = ctx
        _ = self.flags
        LOG.info(
            "Listing registry outputs (domain=%s, materialization=%s, pilot_only=%s)",
            self.domain,
            self.materialization,
            self.pilot_only,
        )

        inventory_path = _inventory_path_or_default(self.inventory_path)
        try:
            inventory = _load_inventory(inventory_path)
        except FileNotFoundError:
            return fail_not_found("inventory", str(inventory_path))
        except ValueError as exc:
            return fail_invalid_value(
                "inventory_path",
                str(inventory_path),
                str(exc),
            )

        filters = _RegistryOutputFilters(
            domain=self.domain,
            materialization=self.materialization,
            targets=set(self.targets) if self.targets else None,
            table_key=self.table_key,
            pilot_only=self.pilot_only,
        )
        filtered = [
            spec
            for spec in inventory.outputs
            if _spec_matches_filters(
                spec,
                filters=filters,
            )
        ]
        filtered.sort(key=lambda spec: (spec.domain, spec.target))

        outputs = [
            DagOutputInfo(
                target=spec.target,
                domain=spec.domain,
                anchor=spec.anchor,
                materialization=spec.materialization,
                table_keys=list(spec.table_keys),
                contracts=list(spec.contracts),
                upstream_targets=list(spec.upstream_targets),
                downstream_consumers=list(spec.downstream_consumers),
                pilot=spec.pilot,
                notes=spec.notes,
            )
            for spec in filtered
        ]

        return CliResult.ok(
            DagOutputListResult(
                outputs=outputs,
                count=len(outputs),
                inventory_path=str(inventory_path),
            )
        )


@cli_command("registry.validate", require_storage=False)
@registry_app.command(name="validate")
@dataclass(frozen=True)
class RegistryValidate(Command[DagOutputValidateResult]):
    """Validate DAG output inventory schema."""

    __operation_id__ = "registry.validate"

    inventory_path: Annotated[
        Path | None,
        option_param(REGISTRY_INVENTORY_PATH, command_path=REGISTRY_VALIDATE_PATH),
    ] = None
    flags: SharedFlagsProtocol = _REGISTRY_VALIDATE_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[DagOutputValidateResult]:
        """Execute registry inventory validation.

        Parameters
        ----------
        ctx
            CLI execution context.

        Returns
        -------
        CliResult[DagOutputValidateResult]
            Registry validation result.
        """
        _ = ctx
        _ = self.flags
        LOG.info("Validating registry inventory")

        inventory_path = _inventory_path_or_default(self.inventory_path)
        try:
            inventory = _load_inventory(inventory_path)
        except FileNotFoundError:
            return fail_not_found("inventory", str(inventory_path))
        except ValueError as exc:
            return fail_invalid_value(
                "inventory_path",
                str(inventory_path),
                str(exc),
            )

        pilot_targets = sorted(spec.target for spec in inventory.outputs if spec.pilot)

        return CliResult.ok(
            DagOutputValidateResult(
                inventory_path=str(inventory_path),
                output_count=len(inventory.outputs),
                pilot_targets=pilot_targets,
                version=inventory.version,
                generated_at=inventory.generated_at,
            )
        )


__all__ = [
    "DagOutputInfo",
    "DagOutputListResult",
    "DagOutputValidateResult",
    "RegistryOutputs",
    "RegistryValidate",
    "registry_app",
]
