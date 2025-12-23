"""Output inventory resolution for build targets."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, cast

from codeintel.build.hamilton.introspect import derive_target_outputs_from_savers
from codeintel.build.hamilton.runtime import HamiltonRuntime
from codeintel.build.output_inventory import OutputInventory
from codeintel.build.settings import get_build_settings
from codeintel.build.target_catalog import load_target_specs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.targets import OutputTarget

OutputInventoryMode = Literal["compare", "dag", "declared"]

log = logging.getLogger(__name__)


def _inventory_from_targets(targets: Iterable[OutputTarget]) -> OutputInventory:
    datasets_by_target: dict[str, tuple[str, ...]] = {}
    artifacts_by_target: dict[str, tuple[str, ...]] = {}
    artifact_templates_by_target: dict[str, dict[str, str]] = {}

    for target in targets:
        datasets_by_target[target.name] = tuple(target.contract.table_keys)
        artifacts_by_target[target.name] = tuple(target.contract.artifact_names)
        artifact_templates_by_target[target.name] = {
            artifact.name: artifact.path_template for artifact in target.contract.artifacts
        }

    return OutputInventory(
        datasets_by_target=datasets_by_target,
        artifacts_by_target=artifacts_by_target,
        artifact_templates_by_target=artifact_templates_by_target,
    )


def _inventory_from_runtime(runtime: HamiltonRuntime) -> OutputInventory:
    return _inventory_from_targets(runtime.graph.all_targets)


@dataclass(frozen=True, slots=True)
class OutputInventoryResolver:
    """Resolve output inventory for build targets."""

    mode: OutputInventoryMode | None = None
    strict: bool | None = None

    def resolve(self, *, runtime: HamiltonRuntime | None = None) -> OutputInventory:
        """Resolve output inventory using configured mode.

        Returns
        -------
        OutputInventory
            Resolved output inventory.
        """
        return resolve_output_inventory(runtime=runtime, mode=self.mode, strict=self.strict)


def _diff_inventories(
    *,
    declared: OutputInventory,
    derived: OutputInventory,
) -> list[str]:
    issues: list[str] = []
    all_targets = set(declared.datasets_by_target) | set(declared.artifacts_by_target)
    all_targets |= set(derived.datasets_by_target) | set(derived.artifacts_by_target)
    all_targets |= set(declared.artifact_templates_by_target)
    all_targets |= set(derived.artifact_templates_by_target)

    for target_name in sorted(all_targets):
        declared_tables = set(declared.datasets_for(target_name))
        declared_artifacts = set(declared.artifacts_for(target_name))
        derived_tables = set(derived.datasets_for(target_name))
        derived_artifacts = set(derived.artifacts_for(target_name))

        if declared_tables != derived_tables:
            issues.append(
                "Target contract table_keys differ from DAG outputs "
                f"for {target_name}: expected={sorted(declared_tables)} observed={sorted(derived_tables)}"
            )
        if declared_artifacts != derived_artifacts:
            issues.append(
                "Target contract artifact_names differ from DAG outputs "
                f"for {target_name}: expected={sorted(declared_artifacts)} observed={sorted(derived_artifacts)}"
            )
        declared_templates = declared.artifact_templates_for(target_name)
        derived_templates = derived.artifact_templates_for(target_name)
        if declared_templates != derived_templates:
            issues.append(
                "Target contract artifact templates differ from DAG outputs "
                f"for {target_name}: expected={declared_templates} observed={derived_templates}"
            )

    return issues


def resolve_output_inventory(
    *,
    runtime: HamiltonRuntime | None = None,
    mode: OutputInventoryMode | None = None,
    strict: bool | None = None,
) -> OutputInventory:
    """Resolve output inventory using the configured mode.

    Parameters
    ----------
    runtime
        Optional Hamilton runtime for DAG-derived inventory.
    mode
        Inventory mode override. Defaults to settings.
    strict
        When True, raise on mismatch between declared and DAG outputs.

    Returns
    -------
    OutputInventory
        Resolved output inventory.

    Raises
    ------
    RuntimeError
        If strict mode is enabled and declared outputs diverge from DAG outputs.
    TypeError
        If the Hamilton driver factory is missing or misconfigured, or DAG inventory
        is requested without a runtime.
    """
    settings = get_build_settings()
    resolved_mode = cast("OutputInventoryMode", mode or settings.output_inventory_source)
    strict_mode = settings.output_inventory_strict if strict is None else strict

    resolved_runtime = runtime
    if resolved_runtime is None and resolved_mode != "declared":
        driver_factory_mod = importlib.import_module("codeintel.build.hamilton.driver_factory")
        build_driver_fn_raw = getattr(driver_factory_mod, "build_driver", None)
        if not callable(build_driver_fn_raw):
            msg = "codeintel.build.hamilton.driver_factory.build_driver is missing or not callable"
            raise TypeError(msg)
        build_driver_fn = cast("Callable[[], HamiltonRuntime]", build_driver_fn_raw)
        resolved_runtime = build_driver_fn()

    declared = (
        _inventory_from_runtime(resolved_runtime)
        if resolved_runtime is not None
        else _inventory_from_targets(load_target_specs())
    )
    if resolved_mode == "declared":
        return declared

    if resolved_runtime is None:
        msg = "DAG inventory requested without a Hamilton runtime"
        raise TypeError(msg)
    derived = derive_target_outputs_from_savers(resolved_runtime)
    dag_inventory = OutputInventory(
        datasets_by_target=derived.datasets_by_target,
        artifacts_by_target=derived.artifacts_by_target,
        artifact_templates_by_target=derived.artifact_templates_by_target,
    )
    issues = _diff_inventories(declared=declared, derived=dag_inventory)
    if issues:
        joined = "\n".join(f"- {issue}" for issue in issues)
        if strict_mode:
            msg = "Output inventory mismatch:\n" + joined
            raise RuntimeError(msg)
        log.warning("Output inventory mismatch:\n%s", joined)

    if resolved_mode == "dag":
        return dag_inventory
    return declared


@lru_cache(maxsize=1)
def get_output_inventory() -> OutputInventory:
    """Return the output inventory derived from target specs.

    Returns
    -------
        OutputInventory
        Output inventory derived from the canonical target catalog.
    """
    return resolve_output_inventory()


def build_output_inventory_snapshot() -> OutputInventory:
    """Build a fresh output inventory snapshot (non-cached).

    Returns
    -------
        OutputInventory
        Output inventory derived from the canonical target catalog.
    """
    return resolve_output_inventory()


__all__ = [
    "OutputInventoryResolver",
    "build_output_inventory_snapshot",
    "get_output_inventory",
    "resolve_output_inventory",
]
