"""Native target registry for Hamilton Phase 3.

This module maintains the registry of native Hamilton targets that have been
migrated from plugin wrappers to pure Hamilton pipelines.

The registry enables progressive migration: targets can be flipped from wrapper
to native implementation without breaking the build system.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from types import ModuleType


@dataclass(frozen=True)
class NativeTargetSpec:
    """Specification for a native Hamilton target.

    Attributes
    ----------
    target_name
        Build target name (e.g., "risk_factors").
    module_path
        Python import path to the native module
        (e.g., "codeintel.build.hamilton.native.analytics.risk_factors").
    """

    target_name: str
    module_path: str


# Registry of native targets
# Wave 1: risk_factors
# Wave 2 PR-21: coverage_functions, hotspots, subsystems
# Wave 2 PR-22: call_graph_views
# Wave 2 PR-23: export_jsonl, export_parquet
# Wave 3 PR-24: scip, typing
NATIVE_TARGETS: Final[tuple[NativeTargetSpec, ...]] = (
    NativeTargetSpec(
        target_name="risk_factors",
        module_path="codeintel.build.hamilton.native.analytics.risk_factors",
    ),
    NativeTargetSpec(
        target_name="coverage_functions",
        module_path="codeintel.build.hamilton.native.analytics.coverage_functions",
    ),
    NativeTargetSpec(
        target_name="hotspots",
        module_path="codeintel.build.hamilton.native.analytics.hotspots",
    ),
    NativeTargetSpec(
        target_name="subsystems",
        module_path="codeintel.build.hamilton.native.analytics.subsystems",
    ),
    NativeTargetSpec(
        target_name="call_graph_views",
        module_path="codeintel.build.hamilton.native.graphs.call_graph_views",
    ),
    NativeTargetSpec(
        target_name="export_jsonl",
        module_path="codeintel.build.hamilton.native.export.export_jsonl",
    ),
    NativeTargetSpec(
        target_name="export_parquet",
        module_path="codeintel.build.hamilton.native.export.export_parquet",
    ),
    NativeTargetSpec(
        target_name="scip",
        module_path="codeintel.build.hamilton.native.ingestion.scip",
    ),
    NativeTargetSpec(
        target_name="typing",
        module_path="codeintel.build.hamilton.native.ingestion.typing",
    ),
)


def native_target_names() -> frozenset[str]:
    """Return the set of target names that have native implementations.

    Returns
    -------
    frozenset[str]
        Set of target names registered as native.

    Examples
    --------
    >>> names = native_target_names()
    >>> "risk_factors" in names  # Will be True after PR-20
    False
    """
    return frozenset(spec.target_name for spec in NATIVE_TARGETS)


def load_native_modules() -> tuple[ModuleType, ...]:
    """Load all native target modules for driver composition.

    Returns
    -------
    tuple[ModuleType, ...]
        Tuple of imported native target modules.

    Raises
    ------
    ImportError
        If a registered module cannot be imported.

    Examples
    --------
    >>> modules = load_native_modules()
    >>> len(modules)  # Will be > 0 after PR-20
    0
    """
    modules: list[ModuleType] = []
    for spec in NATIVE_TARGETS:
        try:
            module = importlib.import_module(spec.module_path)
            modules.append(module)
        except ImportError as e:
            msg = f"Failed to import native target module '{spec.module_path}': {e}"
            raise ImportError(msg) from e
    return tuple(modules)


def is_native_target(target_name: str) -> bool:
    """Check if a target has a native implementation.

    Parameters
    ----------
    target_name
        Build target name to check.

    Returns
    -------
    bool
        True if the target is registered as native.

    Examples
    --------
    >>> is_native_target("risk_factors")  # Will be True after PR-20
    False
    >>> is_native_target("modules")
    False
    """
    return target_name in native_target_names()


__all__ = [
    "NATIVE_TARGETS",
    "NativeTargetSpec",
    "is_native_target",
    "load_native_modules",
    "native_target_names",
]
