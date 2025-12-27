"""Hamilton integration for the CodeIntel build system.

Hamilton is the orchestration layer for CodeIntel's build graph. This package provides:

- Driver construction for native pipelines
- Execution via ``HamiltonBuildExecutor``
- Planning (``compute_plan``) and DAG observability exports
- Contract enforcement hooks for datasets and artifacts

Driver construction
-------------------
The build DAG is composed using native target modules only.

Observability
-------------
This package uses Hamilton lifecycle adapters for telemetry and contract enforcement.

Example
-------
>>> from codeintel.build.hamilton import BuildEnv, HamiltonBuildExecutor
>>> executor = HamiltonBuildExecutor(profile="full")
>>> result = executor.run(env=env, targets=["risk_factors"])
>>> result.success
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import lazy_import

__all__ = [
    "BuildEnv",
    "DatasetRef",
    "HamiltonBuildExecutor",
    "HamiltonBuildPlan",
    "HamiltonBuildResult",
    "HamiltonRuntime",
    "IbisIOConfig",
    "PlanEntry",
    "TargetRunRecord",
    "build_driver",
    "compute_plan",
    "dataset_node",
    "export_dag_json",
    "export_execution_json",
    "get_dag_info",
    "get_pandera_schema",
    "list_available_nodes",
    "list_execution_order",
    "list_execution_targets",
    "refs_from_target_result",
    "target_node",
    "target_to_node_name",
    "to_node_name",
    "validate_dataframe",
    "validate_dataset_ref",
    "with_contract",
]

if TYPE_CHECKING:
    from codeintel.build.hamilton.contracts import (
        get_pandera_schema,
        validate_dataframe,
        validate_dataset_ref,
        with_contract,
    )
    from codeintel.build.hamilton.driver_factory import (
        HamiltonRuntime,
        build_driver,
        list_available_nodes,
        target_to_node_name,
    )
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.executor import HamiltonBuildExecutor, HamiltonBuildResult
    from codeintel.build.hamilton.io import DatasetRef, IbisIOConfig, refs_from_target_result
    from codeintel.build.hamilton.naming import dataset_node, target_node, to_node_name
    from codeintel.build.hamilton.observability import (
        export_dag_json,
        export_execution_json,
        get_dag_info,
        list_execution_order,
        list_execution_targets,
    )
    from codeintel.build.hamilton.planner import (
        HamiltonBuildPlan,
        PlanEntry,
        compute_plan,
    )
    from codeintel.build.hamilton.run_records import TargetRunRecord

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "get_pandera_schema": ("codeintel.build.hamilton.contracts", "get_pandera_schema"),
    "validate_dataframe": ("codeintel.build.hamilton.contracts", "validate_dataframe"),
    "validate_dataset_ref": ("codeintel.build.hamilton.contracts", "validate_dataset_ref"),
    "with_contract": ("codeintel.build.hamilton.contracts", "with_contract"),
    "HamiltonRuntime": ("codeintel.build.hamilton.driver_factory", "HamiltonRuntime"),
    "build_driver": ("codeintel.build.hamilton.driver_factory", "build_driver"),
    "list_available_nodes": ("codeintel.build.hamilton.driver_factory", "list_available_nodes"),
    "target_to_node_name": ("codeintel.build.hamilton.driver_factory", "target_to_node_name"),
    "BuildEnv": ("codeintel.build.hamilton.env", "BuildEnv"),
    "HamiltonBuildExecutor": ("codeintel.build.hamilton.executor", "HamiltonBuildExecutor"),
    "HamiltonBuildResult": ("codeintel.build.hamilton.executor", "HamiltonBuildResult"),
    "DatasetRef": ("codeintel.build.hamilton.io", "DatasetRef"),
    "IbisIOConfig": ("codeintel.build.hamilton.io", "IbisIOConfig"),
    "refs_from_target_result": ("codeintel.build.hamilton.io", "refs_from_target_result"),
    "TargetRunRecord": ("codeintel.build.hamilton.run_records", "TargetRunRecord"),
    "dataset_node": ("codeintel.build.hamilton.naming", "dataset_node"),
    "target_node": ("codeintel.build.hamilton.naming", "target_node"),
    "to_node_name": ("codeintel.build.hamilton.naming", "to_node_name"),
    "export_dag_json": ("codeintel.build.hamilton.observability", "export_dag_json"),
    "export_execution_json": ("codeintel.build.hamilton.observability", "export_execution_json"),
    "get_dag_info": ("codeintel.build.hamilton.observability", "get_dag_info"),
    "list_execution_order": ("codeintel.build.hamilton.observability", "list_execution_order"),
    "list_execution_targets": ("codeintel.build.hamilton.observability", "list_execution_targets"),
    "HamiltonBuildPlan": ("codeintel.build.hamilton.planner", "HamiltonBuildPlan"),
    "PlanEntry": ("codeintel.build.hamilton.planner", "PlanEntry"),
    "compute_plan": ("codeintel.build.hamilton.planner", "compute_plan"),
}


def __getattr__(name: str) -> object:
    """Lazily import Hamilton symbols to avoid import-time cycles.

    Parameters
    ----------
    name
        Attribute name requested from the package.

    Returns
    -------
    object
        The resolved attribute loaded from its defining module.

    Raises
    ------
    AttributeError
        Raised when the requested attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = lazy_import(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
