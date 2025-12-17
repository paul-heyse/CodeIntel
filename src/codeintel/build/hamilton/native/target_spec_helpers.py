"""Helpers for declaring build target metadata alongside native Hamilton nodes.

These helpers support the "DAG-first" target catalog strategy:

- Target *dependencies* are derived from the Hamilton graph.
- Target *metadata* (contracts, resources, execution policy, descriptions) is declared
  next to the native Hamilton materialize nodes, and collected into a catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.contracts import ArtifactSpec, OutputContract
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.resources import DEFAULT_EXECUTION, DEFAULT_RESOURCES
from codeintel.build.schemas.declared_schemas import TABLE_SCHEMAS
from codeintel.build.targets import OutputTarget, TargetModule

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.parameters import TargetParameters
    from codeintel.build.resources import TargetExecution, TargetResources
    from codeintel.core.schemas.primitives import TableSchema


def _resolve_table_schemas(table_keys: Iterable[str]) -> tuple[TableSchema, ...]:
    schemas: list[TableSchema] = []
    for table_key in table_keys:
        schema = TABLE_SCHEMAS.get(table_key)
        if schema is None:
            msg = f"Unknown table schema key: {table_key}"
            raise KeyError(msg)
        schemas.append(schema)
    return tuple(schemas)


def make_output_target(
    *,
    name: str,
    module: TargetModule,
    description: str,
    table_keys: Iterable[str] = (),
    artifacts: Iterable[ArtifactSpec] = (),
    resources: TargetResources = DEFAULT_RESOURCES,
    execution: TargetExecution = DEFAULT_EXECUTION,
    parameters: TargetParameters = EMPTY_PARAMETERS,
) -> OutputTarget:
    """Create an OutputTarget spec with dependencies intentionally empty.

    Parameters
    ----------
    name
        Target name (e.g., "function_metrics").
    module
        Target domain classification (ingestion/graphs/analytics/export).
    description
        Human-readable target description.
    table_keys
        Fully qualified output table keys (schema.table).
    artifacts
        Artifact specs produced by the target.
    resources
        Resource requirements for execution.
    execution
        Execution policy configuration.
    parameters
        Optional tuning parameters for the target.

    Returns
    -------
    OutputTarget
        Target metadata object with contract populated from the schema registry.
    """
    tables = _resolve_table_schemas(table_keys)
    return OutputTarget(
        name=name,
        module=module,
        contract=OutputContract(tables=tables, artifacts=tuple(artifacts)),
        dependencies=(),
        resources=resources,
        execution=execution,
        parameters=parameters,
        description=description,
    )


__all__ = [
    "make_output_target",
]

