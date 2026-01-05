"""Helpers for constructing DagCatalog fixtures in tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from codeintel.build.hamilton.dag_catalog import (
    ArtifactWrite,
    DagCatalog,
    IOSurface,
    NodeDescriptor,
    OutputDescriptor,
    OutputRole,
    TableRead,
    TableWrite,
    TargetDescriptor,
    freeze_mapping,
)
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.resources import DEFAULT_EXECUTION, DEFAULT_RESOURCES
from codeintel.build.targets import TargetModule


@dataclass(frozen=True, slots=True)
class CatalogBuildOptions:
    """Optional overrides for test DagCatalog construction."""

    table_outputs_by_target: Mapping[str, Sequence[OutputDescriptor]] | None = None
    artifact_outputs_by_target: Mapping[str, Sequence[OutputDescriptor]] | None = None
    table_keys_by_target: Mapping[str, Sequence[str]] | None = None
    artifact_specs_by_target: Mapping[str, Sequence[tuple[str, str]]] | None = None
    io_surfaces: Mapping[str, IOSurface] | None = None
    nodes: Mapping[str, NodeDescriptor] | None = None


def make_target_descriptor(
    *,
    name: str,
    module: TargetModule,
    **overrides: object,
) -> TargetDescriptor:
    """Construct a minimal TargetDescriptor for tests.

    Parameters
    ----------
    name
        Target name.
    module
        Target module name.
    overrides
        Optional overrides for dependencies, description, anchor_node, and spec_version.

    Returns
    -------
    TargetDescriptor
        Constructed target descriptor.

    Raises
    ------
    TypeError
        If overrides contain unsupported keys or invalid types.
    """
    dependencies = overrides.pop("dependencies", ())
    if not isinstance(dependencies, tuple) or not all(isinstance(dep, str) for dep in dependencies):
        msg = "dependencies must be tuple[str, ...]"
        raise TypeError(msg)

    description = overrides.pop("description", None)
    if description is not None and not isinstance(description, str):
        msg = "description must be str | None"
        raise TypeError(msg)

    anchor_node = overrides.pop("anchor_node", None)
    if anchor_node is not None and not isinstance(anchor_node, str):
        msg = "anchor_node must be str | None"
        raise TypeError(msg)

    spec_version = overrides.pop("spec_version", "1")
    if not isinstance(spec_version, str):
        msg = "spec_version must be str"
        raise TypeError(msg)

    if overrides:
        unknown = ", ".join(sorted(overrides))
        msg = f"Unknown overrides in make_target_descriptor: {unknown}"
        raise TypeError(msg)

    return TargetDescriptor(
        name=name,
        module=module,
        anchor_node=anchor_node or f"t__{name}",
        dependencies=dependencies,
        resources=DEFAULT_RESOURCES,
        execution=DEFAULT_EXECUTION,
        parameters=EMPTY_PARAMETERS,
        description=description or f"Test target {name}",
        spec_version=spec_version,
    )


def build_catalog(
    *,
    targets: Sequence[TargetDescriptor],
    options: CatalogBuildOptions | None = None,
    **overrides: object,
) -> DagCatalog:
    """Build a minimal DagCatalog for tests.

    Parameters
    ----------
    targets
        Target descriptors to include in the catalog.
    options
        Optional overrides for outputs, IO surfaces, and node metadata.
    **overrides
        Backwards-compatible keyword overrides (same fields as CatalogBuildOptions).

    Returns
    -------
    DagCatalog
        Catalog constructed from the provided descriptors.

    Raises
    ------
    TypeError
        If both options and keyword overrides are supplied, or overrides are unknown.
    """
    if options is not None and overrides:
        msg = "Provide either options or explicit keyword overrides, not both."
        raise TypeError(msg)

    resolved_options = options if options is not None else _options_from_overrides(dict(overrides))
    target_map = {target.name: target for target in targets}
    target_nodes = {target.name: target.anchor_node for target in targets}
    node_to_target = {target.anchor_node: target.name for target in targets}
    target_dependencies = {target.name: tuple(target.dependencies) for target in targets}
    target_dependents = _build_dependents(target_dependencies)

    table_outputs: dict[str, OutputDescriptor] = {}
    artifact_outputs: dict[str, OutputDescriptor] = {}

    table_by_target = _resolve_table_outputs(
        targets=targets,
        table_outputs_by_target=resolved_options.table_outputs_by_target,
        table_keys_by_target=resolved_options.table_keys_by_target,
    )
    artifact_by_target = _resolve_artifact_outputs(
        targets=targets,
        artifact_outputs_by_target=resolved_options.artifact_outputs_by_target,
        artifact_specs_by_target=resolved_options.artifact_specs_by_target,
    )

    for outputs in table_by_target.values():
        for output in outputs:
            table_outputs[output.key] = output
    for outputs in artifact_by_target.values():
        for output in outputs:
            artifact_outputs[output.key] = output

    io_surfaces = resolved_options.io_surfaces
    if io_surfaces is None:
        io_surfaces = {
            target.name: IOSurface(
                target=target.name,
                reads=(),
                table_writes=(),
                artifact_writes=(),
            )
            for target in targets
        }

    nodes = resolved_options.nodes
    if nodes is None:
        nodes = {
            target.anchor_node: NodeDescriptor(
                name=target.anchor_node,
                deps=(),
                tags={},
                tag_spec=None,
            )
            for target in targets
        }

    table_data_nodes: dict[str, str] = {}
    for table_key, output in table_outputs.items():
        data_node = output.tags.get("ci.data_node")
        if isinstance(data_node, str) and data_node:
            table_data_nodes[table_key] = data_node

    return DagCatalog(
        nodes=freeze_mapping(nodes),
        targets=freeze_mapping(target_map),
        target_nodes=freeze_mapping(target_nodes),
        node_to_target=freeze_mapping(node_to_target),
        target_dependencies=freeze_mapping(target_dependencies),
        target_dependents=freeze_mapping(target_dependents),
        table_outputs=freeze_mapping(table_outputs),
        artifact_outputs=freeze_mapping(artifact_outputs),
        table_outputs_by_target=freeze_mapping(table_by_target),
        artifact_outputs_by_target=freeze_mapping(artifact_by_target),
        io_surfaces=freeze_mapping(cast("Mapping[str, IOSurface]", io_surfaces)),
        table_data_nodes=freeze_mapping(table_data_nodes),
    )


def make_io_surface(
    *,
    target: str,
    reads: Sequence[TableRead] = (),
    table_writes: Sequence[TableWrite] = (),
    artifact_writes: Sequence[ArtifactWrite] = (),
) -> IOSurface:
    """Construct a minimal IOSurface for tests.

    Returns
    -------
    IOSurface
        IO surface with the supplied reads and writes.
    """
    return IOSurface(
        target=target,
        reads=tuple(reads),
        table_writes=tuple(table_writes),
        artifact_writes=tuple(artifact_writes),
    )


def _build_dependents(
    dependencies: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    dependents: dict[str, list[str]] = {}
    for target, deps in dependencies.items():
        for dep in deps:
            dependents.setdefault(dep, []).append(target)
    return {name: tuple(sorted(items)) for name, items in dependents.items()}


def make_table_output(
    *,
    table_key: str,
    target: str,
    role: OutputRole = "contract",
    sink: str = "test",
) -> OutputDescriptor:
    """Construct an OutputDescriptor for a table output.

    Returns
    -------
    OutputDescriptor
        Descriptor for the table output.
    """
    return OutputDescriptor(
        kind="table",
        key=table_key,
        role=role,
        producer_target=target,
        saver_node=f"m__table__{table_key}",
        sink=sink,
    )


def make_artifact_output(
    *,
    artifact_name: str,
    target: str,
    path_template: str,
    role: OutputRole = "contract",
    sink: str = "test",
) -> OutputDescriptor:
    """Construct an OutputDescriptor for an artifact output.

    Returns
    -------
    OutputDescriptor
        Descriptor for the artifact output.
    """
    return OutputDescriptor(
        kind="artifact",
        key=artifact_name,
        role=role,
        producer_target=target,
        saver_node=f"m__artifact__{artifact_name}",
        sink=sink,
        artifact_path_template=path_template,
    )


def _resolve_table_outputs(
    *,
    targets: Sequence[TargetDescriptor],
    table_outputs_by_target: Mapping[str, Sequence[OutputDescriptor]] | None,
    table_keys_by_target: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[OutputDescriptor, ...]]:
    if table_outputs_by_target is not None and table_keys_by_target is not None:
        msg = "Provide either table_outputs_by_target or table_keys_by_target, not both."
        raise ValueError(msg)

    target_names = {target.name for target in targets}
    if table_outputs_by_target is not None:
        _validate_output_targets(table_outputs_by_target, target_names, label="table")
        return {name: tuple(outputs) for name, outputs in table_outputs_by_target.items()}

    if table_keys_by_target is not None:
        _validate_output_targets(table_keys_by_target, target_names, label="table")
        return {
            name: tuple(make_table_output(table_key=key, target=name) for key in keys)
            for name, keys in table_keys_by_target.items()
        }

    return dict.fromkeys(target_names, ())


def _resolve_artifact_outputs(
    *,
    targets: Sequence[TargetDescriptor],
    artifact_outputs_by_target: Mapping[str, Sequence[OutputDescriptor]] | None,
    artifact_specs_by_target: Mapping[str, Sequence[tuple[str, str]]] | None,
) -> dict[str, tuple[OutputDescriptor, ...]]:
    if artifact_outputs_by_target is not None and artifact_specs_by_target is not None:
        msg = "Provide either artifact_outputs_by_target or artifact_specs_by_target, not both."
        raise ValueError(msg)

    target_names = {target.name for target in targets}
    if artifact_outputs_by_target is not None:
        _validate_output_targets(artifact_outputs_by_target, target_names, label="artifact")
        return {name: tuple(outputs) for name, outputs in artifact_outputs_by_target.items()}

    if artifact_specs_by_target is not None:
        _validate_output_targets(artifact_specs_by_target, target_names, label="artifact")
        return {
            name: tuple(
                make_artifact_output(
                    artifact_name=spec[0],
                    path_template=spec[1],
                    target=name,
                )
                for spec in specs
            )
            for name, specs in artifact_specs_by_target.items()
        }

    return dict.fromkeys(target_names, ())


def _validate_output_targets(
    output_map: Mapping[str, Sequence[object]],
    target_names: set[str],
    *,
    label: str,
) -> None:
    unknown = sorted(set(output_map) - target_names)
    if unknown:
        msg = f"Unknown {label} output targets: {unknown}"
        raise ValueError(msg)


def _options_from_overrides(overrides: dict[str, object]) -> CatalogBuildOptions:
    table_outputs_by_target = cast(
        "Mapping[str, Sequence[OutputDescriptor]] | None",
        overrides.pop("table_outputs_by_target", None),
    )
    artifact_outputs_by_target = cast(
        "Mapping[str, Sequence[OutputDescriptor]] | None",
        overrides.pop("artifact_outputs_by_target", None),
    )
    table_keys_by_target = cast(
        "Mapping[str, Sequence[str]] | None",
        overrides.pop("table_keys_by_target", None),
    )
    artifact_specs_by_target = cast(
        "Mapping[str, Sequence[tuple[str, str]]] | None",
        overrides.pop("artifact_specs_by_target", None),
    )
    io_surfaces = cast("Mapping[str, IOSurface] | None", overrides.pop("io_surfaces", None))
    nodes = cast("Mapping[str, NodeDescriptor] | None", overrides.pop("nodes", None))

    if overrides:
        unknown = ", ".join(sorted(overrides))
        msg = f"Unknown overrides in build_catalog: {unknown}"
        raise TypeError(msg)

    return CatalogBuildOptions(
        table_outputs_by_target=table_outputs_by_target,
        artifact_outputs_by_target=artifact_outputs_by_target,
        table_keys_by_target=table_keys_by_target,
        artifact_specs_by_target=artifact_specs_by_target,
        io_surfaces=io_surfaces,
        nodes=nodes,
    )


__all__ = [
    "build_catalog",
    "make_artifact_output",
    "make_io_surface",
    "make_table_output",
    "make_target_descriptor",
]
