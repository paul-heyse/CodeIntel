"""DAG-free output inventory derived from target specs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.target_catalog import load_target_specs
from codeintel.build.target_metadata import OutputInventory

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.targets import OutputTarget


def _inventory_from_targets(targets: Iterable[OutputTarget]) -> OutputInventory:
    datasets_by_target: dict[str, tuple[str, ...]] = {}
    artifacts_by_target: dict[str, tuple[str, ...]] = {}

    for target in targets:
        datasets_by_target[target.name] = tuple(target.contract.table_keys)
        artifacts_by_target[target.name] = tuple(target.contract.artifact_names)

    return OutputInventory(
        datasets_by_target=datasets_by_target,
        artifacts_by_target=artifacts_by_target,
    )


@lru_cache(maxsize=1)
def get_output_inventory() -> OutputInventory:
    """Return the output inventory derived from target specs.

    Returns
    -------
    OutputInventory
        DAG-free output inventory derived from native target specs.
    """
    return _inventory_from_targets(load_target_specs())


def build_output_inventory_snapshot() -> OutputInventory:
    """Build a fresh output inventory snapshot (non-cached).

    Returns
    -------
    OutputInventory
        Output inventory derived from current target specs.
    """
    return _inventory_from_targets(load_target_specs())


__all__ = [
    "build_output_inventory_snapshot",
    "get_output_inventory",
]
