"""Helpers for declaring build target metadata alongside native Hamilton nodes.

These helpers support the "DAG-first" target catalog strategy:

- Target *dependencies* are derived from the Hamilton graph.
- Target *metadata* (contracts, resources, execution policy, descriptions) is declared
  next to the native Hamilton materialize nodes, and collected into a catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.contracts import OutputContract, placeholder_table_schema
from codeintel.build.hamilton.materializers.path_templates import validate_path_template
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.resources import DEFAULT_EXECUTION, DEFAULT_RESOURCES
from codeintel.build.targets import OutputTarget
from codeintel.storage.helpers.table_key import validate_table_key

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.contracts import ArtifactSpec
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.resources import TargetExecution, TargetResources
    from codeintel.build.targets import TargetModule
    from codeintel.core.schemas.primitives import TableSchema


_TARGET_REGISTRY: dict[str, OutputTarget] = {}


def _validate_table_key(table_key: str) -> None:
    if not table_key:
        msg = "table_key must be non-empty"
        raise ValueError(msg)
    validate_table_key(table_key)


def _validate_artifact_specs(artifacts: tuple[ArtifactSpec, ...]) -> None:
    seen_names: set[str] = set()

    for artifact in artifacts:
        if not artifact.name:
            msg = "artifact.name must be non-empty"
            raise ValueError(msg)
        if artifact.name in seen_names:
            msg = f"Duplicate artifact name in target spec: {artifact.name}"
            raise ValueError(msg)
        seen_names.add(artifact.name)

        if artifact.path_template is not None:
            validate_path_template(artifact.path_template)


def _resolve_table_schemas(
    table_keys: Iterable[str],
    override_tables: Iterable[TableSchema],
) -> tuple[TableSchema, ...]:
    schemas: list[TableSchema] = []
    seen: set[str] = set()
    overrides: dict[str, TableSchema] = {}

    for table_schema in override_tables:
        _validate_table_key(table_schema.table_key)
        if table_schema.table_key in overrides:
            msg = f"Duplicate override table schema: {table_schema.table_key}"
            raise ValueError(msg)
        overrides[table_schema.table_key] = table_schema

    for table_key in table_keys:
        _validate_table_key(table_key)
        if table_key in seen:
            msg = f"Duplicate table_key in target spec: {table_key}"
            raise ValueError(msg)
        seen.add(table_key)
        override_schema = overrides.get(table_key)
        if override_schema is not None:
            schemas.append(override_schema)
            continue

        schemas.append(placeholder_table_schema(table_key))

    extra_overrides = sorted(set(overrides) - seen)
    if extra_overrides:
        msg = f"Override tables not declared in table_keys: {extra_overrides}"
        raise ValueError(msg)

    return tuple(schemas)


@dataclass(frozen=True)
class TargetSpecOptions:
    """Options for declaring a target spec.

    Parameters
    ----------
    table_keys
        Fully qualified table keys produced by the target (schema.table).
    override_tables
        Explicit TableSchema overrides for non-inferable outputs.
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
    override_tables: tuple[TableSchema, ...] = ()
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
        tables = _resolve_table_schemas(
            resolved.table_keys,
            resolved.override_tables,
        )
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


def register_output_target(target: OutputTarget) -> None:
    """Register OutputTarget metadata for Hamilton target discovery.

    Parameters
    ----------
    target
        OutputTarget metadata to register.

    Raises
    ------
    ValueError
        If a target name is already registered.
    """
    existing = _TARGET_REGISTRY.get(target.name)
    if existing is not None:
        msg = f"Duplicate OutputTarget registration: {target.name}"
        raise ValueError(msg)
    _TARGET_REGISTRY[target.name] = target


def register_output_targets(*targets: OutputTarget) -> None:
    """Register multiple OutputTarget metadata entries."""
    for target in targets:
        register_output_target(target)


def get_registered_target(name: str) -> OutputTarget | None:
    """Return a registered OutputTarget by name.

    Returns
    -------
    OutputTarget | None
        Registered target, or None when not present.
    """
    return _TARGET_REGISTRY.get(name)


def resolve_registered_targets(target_names: Iterable[str]) -> tuple[OutputTarget, ...]:
    """Resolve registered OutputTargets for a set of target names.

    Returns
    -------
    tuple[OutputTarget, ...]
        Registered targets aligned with the supplied names.

    Raises
    ------
    RuntimeError
        If registered targets are missing or extra relative to the Hamilton DAG.
    """
    resolved: list[OutputTarget] = []
    missing: list[str] = []
    for name in sorted(target_names):
        target = _TARGET_REGISTRY.get(name)
        if target is None:
            missing.append(name)
            continue
        resolved.append(target)
    if missing:
        msg = f"Missing OutputTarget metadata for targets: {', '.join(missing)}"
        raise RuntimeError(msg)
    extra = sorted(set(_TARGET_REGISTRY) - set(target_names))
    if extra:
        msg = f"Registered OutputTarget metadata not present in Hamilton DAG: {', '.join(extra)}"
        raise RuntimeError(msg)
    return tuple(resolved)


def clear_target_registry() -> None:
    """Clear the registered OutputTarget metadata."""
    _TARGET_REGISTRY.clear()


__all__ = [
    "TargetSpecOptions",
    "clear_target_registry",
    "get_registered_target",
    "make_output_target",
    "register_output_target",
    "register_output_targets",
    "resolve_registered_targets",
]
