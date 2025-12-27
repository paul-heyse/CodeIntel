"""Helpers for constructing DagCatalog fixtures in tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.dag_catalog import (
    ArtifactWrite,
    DagCatalog,
    IOSurface,
    NodeDescriptor,
    OutputDescriptor,
    TableRead,
    TableWrite,
    TargetDescriptor,
    freeze_mapping,
)
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.resources import DEFAULT_EXECUTION, DEFAULT_RESOURCES
from codeintel.build.targets import TargetModule


def make_target_descriptor(
    *,
    name: str,
    module: TargetModule,
    contract: OutputContract,
    **overrides: object,
) -> TargetDescriptor:
    """Construct a minimal TargetDescriptor for tests.

    Parameters
    ----------
    name
        Target name.
    module
        Target module name.
    contract
        Output contract for the target.
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
    if not isinstance(dependencies, tuple) or not all(
        isinstance(dep, str) for dep in dependencies
    ):
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
        contract=contract,
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
    table_outputs_by_target: Mapping[str, Sequence[OutputDescriptor]] | None = None,
    artifact_outputs_by_target: Mapping[str, Sequence[OutputDescriptor]] | None = None,
    io_surfaces: Mapping[str, IOSurface] | None = None,
    nodes: Mapping[str, NodeDescriptor] | None = None,
) -> DagCatalog:
    """Build a minimal DagCatalog for tests.

    Returns
    -------
    DagCatalog
        Catalog constructed from the provided descriptors.
    """
    target_map = {target.name: target for target in targets}
    target_nodes = {target.name: target.anchor_node for target in targets}
    node_to_target = {target.anchor_node: target.name for target in targets}
    target_dependencies = {target.name: tuple(target.dependencies) for target in targets}
    target_dependents = _build_dependents(target_dependencies)

    table_outputs: dict[str, OutputDescriptor] = {}
    artifact_outputs: dict[str, OutputDescriptor] = {}

    table_by_target = {
        name: tuple(outputs)
        for name, outputs in (table_outputs_by_target or {}).items()
    }
    artifact_by_target = {
        name: tuple(outputs)
        for name, outputs in (artifact_outputs_by_target or {}).items()
    }

    for outputs in table_by_target.values():
        for output in outputs:
            table_outputs[output.key] = output
    for outputs in artifact_by_target.values():
        for output in outputs:
            artifact_outputs[output.key] = output

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
