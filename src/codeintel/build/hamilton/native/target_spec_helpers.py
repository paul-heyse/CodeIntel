"""Helpers for declaring build target metadata alongside native Hamilton nodes.

These helpers support the "DAG-first" target catalog strategy:

- Target *dependencies* are derived from the Hamilton graph.
- Target *metadata* (contracts, resources, execution policy, descriptions) is declared
  next to the native Hamilton materialize nodes, and collected into a catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import TYPE_CHECKING

from codeintel.build.contracts import OutputContract
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.resources import DEFAULT_EXECUTION, DEFAULT_RESOURCES
from codeintel.build.targets import OutputTarget
from codeintel.config.datasets.declared_schemas import TABLE_SCHEMAS
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.contracts import ArtifactSpec
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.resources import TargetExecution, TargetResources
    from codeintel.build.targets import TargetModule
    from codeintel.core.schemas.primitives import TableSchema


_ALLOWED_ARTIFACT_TEMPLATE_KEYS: frozenset[str] = frozenset(
    {
        "build_dir",
        "export_dir",
        "repo_root",
        "scip_dir",
    }
)


def _validate_table_key(table_key: str) -> None:
    if not table_key:
        msg = "table_key must be non-empty"
        raise ValueError(msg)
    if "." not in table_key:
        msg = f"table_key must be fully-qualified 'schema.table', got {table_key!r}"
        raise ValueError(msg)

    schema, table = split_table_key(table_key)
    if not schema or not table:
        msg = f"table_key must be fully-qualified 'schema.table', got {table_key!r}"
        raise ValueError(msg)

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if not set(schema) <= allowed or not set(table) <= allowed:
        msg = f"table_key must be alphanumeric/underscore, got {table_key!r}"
        raise ValueError(msg)


def _validate_artifact_specs(artifacts: tuple[ArtifactSpec, ...]) -> None:
    seen_names: set[str] = set()
    formatter = Formatter()

    for artifact in artifacts:
        if not artifact.name:
            msg = "artifact.name must be non-empty"
            raise ValueError(msg)
        if artifact.name in seen_names:
            msg = f"Duplicate artifact name in target spec: {artifact.name}"
            raise ValueError(msg)
        seen_names.add(artifact.name)

        if artifact.path_template is None:
            continue

        for _, field_name, _, _ in formatter.parse(artifact.path_template):
            if field_name is None:
                continue
            if field_name not in _ALLOWED_ARTIFACT_TEMPLATE_KEYS:
                msg = (
                    "Unsupported artifact path_template placeholder "
                    f"{field_name!r} (allowed={sorted(_ALLOWED_ARTIFACT_TEMPLATE_KEYS)})"
                )
                raise ValueError(msg)


def _resolve_table_schemas(table_keys: Iterable[str]) -> tuple[TableSchema, ...]:
    schemas: list[TableSchema] = []
    seen: set[str] = set()
    for table_key in table_keys:
        _validate_table_key(table_key)
        if table_key in seen:
            msg = f"Duplicate table_key in target spec: {table_key}"
            raise ValueError(msg)
        seen.add(table_key)
        schema = TABLE_SCHEMAS.get(table_key)
        if schema is None:
            msg = f"Unknown table schema key: {table_key}"
            raise KeyError(msg)
        schemas.append(schema)
    return tuple(schemas)


@dataclass(frozen=True)
class TargetSpecOptions:
    """Options for declaring a target spec.

    Parameters
    ----------
    table_keys
        Fully qualified table keys produced by the target (schema.table).
    artifacts
        Artifact specs produced by the target.
    resources
        Resource requirements for execution.
    execution
        Execution policy configuration.
    parameters
        Optional tuning parameters for the target.
    """

    table_keys: tuple[str, ...] = ()
    artifacts: tuple[ArtifactSpec, ...] = ()
    resources: TargetResources = DEFAULT_RESOURCES
    execution: TargetExecution = DEFAULT_EXECUTION
    parameters: TargetParameters = EMPTY_PARAMETERS


def make_output_target(
    *,
    name: str,
    module: TargetModule,
    description: str,
    options: TargetSpecOptions | None = None,
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
    options
        Optional TargetSpecOptions bundle for tables/artifacts/resources/execution/parameters.

    Returns
    -------
    OutputTarget
        Target metadata object with contract populated from the schema registry.

    Raises
    ------
    ValueError
        If the target spec is invalid (missing name/description, invalid/duplicate table keys, or
        invalid artifact specs).
    """
    if not name:
        msg = "Target name must be non-empty"
        raise ValueError(msg)
    if not description:
        msg = f"Target {name} must provide a non-empty description"
        raise ValueError(msg)

    resolved = TargetSpecOptions() if options is None else options
    _validate_artifact_specs(resolved.artifacts)
    try:
        tables = _resolve_table_schemas(resolved.table_keys)
    except KeyError as exc:
        msg = f"Unknown table schema key in target spec {name}: {exc}"
        raise ValueError(msg) from exc
    return OutputTarget(
        name=name,
        module=module,
        contract=OutputContract(tables=tables, artifacts=resolved.artifacts),
        dependencies=(),
        resources=resolved.resources,
        execution=resolved.execution,
        parameters=resolved.parameters,
        description=description,
    )


__all__ = [
    "TargetSpecOptions",
    "make_output_target",
]
