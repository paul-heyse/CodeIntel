"""Compile DagCatalog from a Hamilton Driver graph."""

from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import FunctionType, MappingProxyType, MethodType
from typing import TYPE_CHECKING, TypeGuard, cast

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
from codeintel.build.hamilton.tag_spec import TagSpec, TagValue, tag_spec_from_tags
from codeintel.build.hamilton.target_spec_compiler import (
    TargetDescriptorCompileInputs,
    TargetSpecOverride,
    compile_target_descriptors_from_driver,
)
from codeintel.core.hamilton import tags as ht
from codeintel.core.table_key import TableKeyValidationError, validate_table_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hamilton.driver import Driver
    from hamilton.node import Node


@dataclass(frozen=True, slots=True)
class _OutputInventory:
    table_outputs: dict[str, OutputDescriptor]
    artifact_outputs: dict[str, OutputDescriptor]
    table_outputs_by_target: dict[str, tuple[OutputDescriptor, ...]]
    artifact_outputs_by_target: dict[str, tuple[OutputDescriptor, ...]]


def _attach_output_metadata(
    targets: Sequence[TargetDescriptor],
    *,
    outputs: _OutputInventory,
) -> tuple[TargetDescriptor, ...]:
    enriched: list[TargetDescriptor] = []
    for target in targets:
        table_keys = tuple(
            sorted(output.key for output in outputs.table_outputs_by_target.get(target.name, ()))
        )
        artifact_names = tuple(
            sorted(output.key for output in outputs.artifact_outputs_by_target.get(target.name, ()))
        )
        enriched.append(
            replace(
                target,
                table_keys=table_keys,
                artifact_names=artifact_names,
            )
        )
    return tuple(enriched)


def compile_dag_catalog(
    driver: Driver,
    *,
    overrides_by_target: Mapping[str, TargetSpecOverride] | None = None,
    strict: bool = True,
) -> DagCatalog:
    """Compile a DagCatalog from a Hamilton Driver graph.

    Parameters
    ----------
    driver
        Hamilton Driver with a built FunctionGraph.
    overrides_by_target
        Optional target spec overrides (resources/execution/parameters).
    strict
        When True, enforce invariants and raise on missing/invalid tags.

    Returns
    -------
    DagCatalog
        Immutable catalog derived from Hamilton graph + tags.
    """
    nodes = _compile_nodes(driver.graph.nodes)
    target_nodes, node_to_target = _collect_target_nodes(nodes, strict=strict)
    deps_by_target = _derive_target_dependencies(nodes, target_nodes, node_to_target)
    outputs = _compile_output_inventory(driver.graph.nodes, strict=strict)

    resolved_overrides = overrides_by_target or MappingProxyType({})
    inputs = TargetDescriptorCompileInputs(
        overrides_by_target=resolved_overrides,
        deps_by_target=deps_by_target,
    )
    targets = compile_target_descriptors_from_driver(
        driver,
        inputs=inputs,
        strict=strict,
    )
    targets_with_outputs = _attach_output_metadata(targets, outputs=outputs)
    target_map = {target.name: target for target in targets_with_outputs}

    _validate_target_nodes(target_map, target_nodes, strict=strict)
    dependents = _build_dependents_map(deps_by_target)
    io_surfaces = _compile_io_surfaces(
        nodes=nodes,
        target_nodes=target_nodes,
        node_to_target=node_to_target,
        outputs=outputs,
    )

    return DagCatalog(
        nodes=freeze_mapping(nodes),
        targets=freeze_mapping(target_map),
        target_nodes=freeze_mapping(target_nodes),
        node_to_target=freeze_mapping(node_to_target),
        target_dependencies=freeze_mapping(deps_by_target),
        target_dependents=freeze_mapping(dependents),
        table_outputs=freeze_mapping(outputs.table_outputs),
        artifact_outputs=freeze_mapping(outputs.artifact_outputs),
        table_outputs_by_target=freeze_mapping(outputs.table_outputs_by_target),
        artifact_outputs_by_target=freeze_mapping(outputs.artifact_outputs_by_target),
        io_surfaces=freeze_mapping(io_surfaces),
    )


def _compile_nodes(nodes: Mapping[str, Node]) -> dict[str, NodeDescriptor]:
    compiled: dict[str, NodeDescriptor] = {}
    for name, node in nodes.items():
        tags = _node_tags(node)
        tag_spec = _parse_tag_spec(tags)
        deps = tuple(dep.name for dep in node.dependencies)
        module_name, module_path = _resolve_node_module(node)
        compiled[name] = NodeDescriptor(
            name=name,
            deps=deps,
            tags=MappingProxyType(tags),
            tag_spec=tag_spec,
            module=module_name,
            module_path=module_path,
        )
    return compiled


def _resolve_node_module(node: Node) -> tuple[str | None, Path | None]:
    candidate = getattr(node, "callable", None)
    if candidate is None:
        candidate = getattr(node, "callabl", None)
    if isinstance(candidate, (FunctionType, MethodType)):
        module_name = getattr(candidate, "__module__", None)
    else:
        module_name = getattr(candidate, "__module__", None) if callable(candidate) else None
    module_path = None
    if module_name and callable(candidate):
        try:
            source_path = inspect.getsourcefile(candidate)
        except (OSError, TypeError):
            source_path = None
        if source_path:
            module_path = Path(source_path).resolve()
    return module_name if isinstance(module_name, str) else None, module_path


def _node_tags(node: Node) -> dict[str, object]:
    tags = getattr(node, "tags", None)
    if isinstance(tags, dict):
        return dict(tags)
    return {}


def _parse_tag_spec(tags: Mapping[str, object]) -> TagSpec | None:
    return tag_spec_from_tags(cast("Mapping[str, TagValue]", tags))


def _collect_target_nodes(
    nodes: Mapping[str, NodeDescriptor],
    *,
    strict: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    target_nodes: dict[str, str] = {}
    node_to_target: dict[str, str] = {}

    for node in nodes.values():
        tags = node.tags
        if tags.get(ht.TAG_NODE_TYPE) != ht.NODE_TYPE_MATERIALIZE:
            continue
        target = tags.get(ht.TAG_TARGET)
        if not isinstance(target, str) or not target:
            if strict:
                msg = f"Materialize node {node.name} missing target tag"
                raise RuntimeError(msg)
            continue
        if target in target_nodes:
            msg = f"Duplicate materialize nodes for target '{target}'"
            raise RuntimeError(msg)
        target_nodes[target] = node.name
        node_to_target[node.name] = target

    if strict and not target_nodes:
        msg = "No target anchors discovered in Hamilton graph"
        raise RuntimeError(msg)

    return target_nodes, node_to_target


def _derive_target_dependencies(
    nodes: Mapping[str, NodeDescriptor],
    target_nodes: Mapping[str, str],
    node_to_target: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    deps_by_target: dict[str, tuple[str, ...]] = {}
    for target_name, node_name in target_nodes.items():
        deps = _direct_target_dependencies(
            nodes=nodes,
            node_to_target=node_to_target,
            root_node=node_name,
        )
        deps_by_target[target_name] = tuple(sorted(deps))
    return deps_by_target


def _direct_target_dependencies(
    *,
    nodes: Mapping[str, NodeDescriptor],
    node_to_target: Mapping[str, str],
    root_node: str,
) -> frozenset[str]:
    root_target = node_to_target.get(root_node)
    if root_target is None:
        msg = f"Root node is not a materialize node: {root_node}"
        raise RuntimeError(msg)

    deps: set[str] = set()
    visited: set[str] = set()
    stack = list(nodes[root_node].deps)

    while stack:
        node_name = stack.pop()
        if node_name in visited:
            continue
        visited.add(node_name)

        target = node_to_target.get(node_name)
        if target is not None:
            if target != root_target:
                deps.add(target)
            continue

        node = nodes.get(node_name)
        if node is None:
            continue
        stack.extend(node.deps)

    return frozenset(deps)


def _compile_output_inventory(
    nodes: Mapping[str, Node],
    *,
    strict: bool,
) -> _OutputInventory:
    table_outputs: dict[str, OutputDescriptor] = {}
    artifact_outputs: dict[str, OutputDescriptor] = {}
    table_by_target: dict[str, list[OutputDescriptor]] = {}
    artifact_by_target: dict[str, list[OutputDescriptor]] = {}

    for node_name, node in nodes.items():
        output = _output_from_node(node_name=node_name, node=node, strict=strict)
        if output is None:
            continue
        if output.kind == "table":
            _record_table_output(
                output=output,
                table_outputs=table_outputs,
                table_by_target=table_by_target,
            )
        else:
            _record_artifact_output(
                output=output,
                artifact_outputs=artifact_outputs,
                artifact_by_target=artifact_by_target,
            )

    return _OutputInventory(
        table_outputs=table_outputs,
        artifact_outputs=artifact_outputs,
        table_outputs_by_target={k: tuple(v) for k, v in table_by_target.items()},
        artifact_outputs_by_target={k: tuple(v) for k, v in artifact_by_target.items()},
    )


def _output_from_node(
    *,
    node_name: str,
    node: Node,
    strict: bool,
) -> OutputDescriptor | None:
    tags = _node_tags(node)
    if tags.get("hamilton.data_saver") is not True:
        return None
    output_role = _require_output_role(tags=tags, node_name=node_name, strict=strict)
    target = _require_tag(
        tags=tags,
        node_name=node_name,
        key=ht.TAG_TARGET,
        label="target",
        strict=strict,
    )
    table_key, artifact_name = _resolve_output_identity(
        tags=tags,
        node_name=node_name,
        strict=strict,
    )
    missing_identity = table_key is None and artifact_name is None
    if output_role is None or target is None or missing_identity:
        return None

    sink = tags.get("hamilton.data_saver.sink")
    sink_str = sink if isinstance(sink, str) and sink else "unknown"

    output_tags = MappingProxyType(tags)
    output: OutputDescriptor | None = None
    if table_key is not None:
        _validate_table_key(table_key=table_key, node_name=node_name, strict=strict)
        output = OutputDescriptor(
            kind="table",
            key=table_key,
            role=output_role,
            producer_target=target,
            saver_node=node_name,
            sink=sink_str,
            tags=output_tags,
        )
    elif artifact_name is not None:
        template = _require_tag(
            tags=tags,
            node_name=node_name,
            key=ht.TAG_ARTIFACT_PATH_TEMPLATE,
            label="artifact_path_template",
            strict=strict,
        )
        if template is not None:
            output = OutputDescriptor(
                kind="artifact",
                key=artifact_name,
                role=output_role,
                producer_target=target,
                saver_node=node_name,
                sink=sink_str,
                artifact_path_template=template,
                tags=output_tags,
            )

    return output


def _validate_table_key(*, table_key: str, node_name: str, strict: bool) -> None:
    try:
        validate_table_key(table_key)
    except TableKeyValidationError as exc:
        if strict:
            msg = f"DataSaver node {node_name} has invalid table_key: {table_key}"
            raise RuntimeError(msg) from exc


def _compatible_table_output(existing: OutputDescriptor, candidate: OutputDescriptor) -> bool:
    return (
        existing.kind == candidate.kind
        and existing.key == candidate.key
        and existing.role == candidate.role
        and existing.sink == candidate.sink
        and existing.artifact_path_template == candidate.artifact_path_template
    )


def _table_output_sort_key(output: OutputDescriptor) -> tuple[str, str]:
    return (output.producer_target, output.saver_node)


def _prefer_table_output(left: OutputDescriptor, right: OutputDescriptor) -> OutputDescriptor:
    return min(left, right, key=_table_output_sort_key)


def _record_table_output(
    *,
    output: OutputDescriptor,
    table_outputs: dict[str, OutputDescriptor],
    table_by_target: dict[str, list[OutputDescriptor]],
) -> None:
    if output.role != "contract":
        return
    existing = table_outputs.get(output.key)
    if existing is None:
        table_outputs[output.key] = output
    else:
        if not _compatible_table_output(existing, output):
            msg = (
                f"Duplicate contract table output: {output.key} "
                f"({existing.producer_target} vs {output.producer_target})"
            )
            raise RuntimeError(msg)
        table_outputs[output.key] = _prefer_table_output(existing, output)
    table_by_target.setdefault(output.producer_target, []).append(output)


def _record_artifact_output(
    *,
    output: OutputDescriptor,
    artifact_outputs: dict[str, OutputDescriptor],
    artifact_by_target: dict[str, list[OutputDescriptor]],
) -> None:
    if output.role != "contract":
        return
    if output.key in artifact_outputs:
        msg = f"Duplicate contract artifact output: {output.key}"
        raise RuntimeError(msg)
    artifact_outputs[output.key] = output
    artifact_by_target.setdefault(output.producer_target, []).append(output)


def _require_output_role(
    *,
    tags: Mapping[str, object],
    node_name: str,
    strict: bool,
) -> OutputRole | None:
    output_role = tags.get("output_role")
    if _is_output_role(output_role):
        return output_role
    if strict:
        msg = f"DataSaver node {node_name} missing/invalid output_role tag"
        raise RuntimeError(msg)
    return None


_OUTPUT_ROLE_VALUES: frozenset[OutputRole] = frozenset(("contract", "internal"))


def _is_output_role(value: object) -> TypeGuard[OutputRole]:
    return isinstance(value, str) and value in _OUTPUT_ROLE_VALUES


def _resolve_output_identity(
    *,
    tags: Mapping[str, object],
    node_name: str,
    strict: bool,
) -> tuple[str | None, str | None]:
    table_key = _optional_tag(tags=tags, key=ht.TAG_TABLE_KEY)
    artifact_name = _optional_tag(tags=tags, key=ht.TAG_ARTIFACT)
    if (table_key is None) == (artifact_name is None):
        if strict:
            msg = f"DataSaver node {node_name} missing table_key/artifact tags"
            raise RuntimeError(msg)
        return None, None
    return table_key, artifact_name


def _require_tag(
    *,
    tags: Mapping[str, object],
    node_name: str,
    key: str,
    label: str,
    strict: bool,
) -> str | None:
    value = tags.get(key)
    if isinstance(value, str) and value:
        return value
    if strict:
        msg = f"DataSaver node {node_name} missing {label} tag"
        raise RuntimeError(msg)
    return None


def _optional_tag(*, tags: Mapping[str, object], key: str) -> str | None:
    value = tags.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _build_dependents_map(
    deps_by_target: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    dependents: dict[str, list[str]] = {}
    for target_name, deps in deps_by_target.items():
        for dep in deps:
            dependents.setdefault(dep, []).append(target_name)
        dependents.setdefault(target_name, dependents.get(target_name, []))
    return {k: tuple(sorted(v)) for k, v in dependents.items()}


def _compile_io_surfaces(
    *,
    nodes: Mapping[str, NodeDescriptor],
    target_nodes: Mapping[str, str],
    node_to_target: Mapping[str, str],
    outputs: _OutputInventory,
) -> dict[str, IOSurface]:
    surfaces: dict[str, IOSurface] = {}
    for target_name, node_name in target_nodes.items():
        root = nodes.get(node_name)
        if root is None:
            continue
        reads = _collect_target_reads(
            nodes=nodes,
            root=root,
            node_to_target=node_to_target,
            target_name=target_name,
        )
        table_writes = _table_writes_for_target(outputs, target_name=target_name)
        artifact_writes = _artifact_writes_for_target(outputs, target_name=target_name)

        reads_deduped = {
            (r.table_key, r.loader_type, r.producer_target, r.loader_node): r for r in reads
        }
        surfaces[target_name] = IOSurface(
            target=target_name,
            reads=tuple(
                sorted(
                    reads_deduped.values(),
                    key=lambda r: (
                        r.table_key,
                        r.loader_type,
                        r.producer_target or "",
                        r.loader_node,
                    ),
                )
            ),
            table_writes=tuple(
                sorted(
                    table_writes,
                    key=lambda w: (w.table_key, w.sink, w.saver_node),
                )
            ),
            artifact_writes=tuple(
                sorted(
                    artifact_writes,
                    key=lambda w: (w.artifact_name, w.sink, w.saver_node),
                )
            ),
        )
    return surfaces


def _table_writes_for_target(
    outputs: _OutputInventory,
    *,
    target_name: str,
) -> list[TableWrite]:
    return [
        TableWrite(
            table_key=output.key,
            sink=output.sink,
            saver_node=output.saver_node,
        )
        for output in outputs.table_outputs_by_target.get(target_name, ())
    ]


def _artifact_writes_for_target(
    outputs: _OutputInventory,
    *,
    target_name: str,
) -> list[ArtifactWrite]:
    return [
        ArtifactWrite(
            artifact_name=output.key,
            sink=output.sink,
            saver_node=output.saver_node,
        )
        for output in outputs.artifact_outputs_by_target.get(target_name, ())
    ]


def _collect_target_reads(
    *,
    nodes: Mapping[str, NodeDescriptor],
    root: NodeDescriptor,
    node_to_target: Mapping[str, str],
    target_name: str,
) -> list[TableRead]:
    reads: list[TableRead] = []
    seen: set[str] = set()
    queue: deque[str] = deque(root.deps)

    while queue:
        node_name = queue.popleft()
        if node_name in seen:
            continue
        seen.add(node_name)

        upstream_target = node_to_target.get(node_name)
        if upstream_target is not None and upstream_target != target_name:
            continue

        node = nodes.get(node_name)
        if node is None:
            continue
        node_type = node.tags.get(ht.TAG_NODE_TYPE)

        read = _read_from_node(node=node, node_type=node_type)
        if read is not None:
            reads.append(read)
            continue

        if node_type == ht.NODE_TYPE_MATERIALIZE and node.name != root.name:
            continue

        queue.extend(node.deps)

    return reads


def _read_from_node(
    *,
    node: NodeDescriptor,
    node_type: object,
) -> TableRead | None:
    if node_type not in {ht.NODE_TYPE_LOADER_QUERY, ht.NODE_TYPE_DATASET}:
        return None

    table_key = node.tags.get(ht.TAG_TABLE_KEY)
    if not isinstance(table_key, str) or not table_key:
        return None

    producer = node.tags.get(ht.TAG_TARGET)
    producer_str = producer if isinstance(producer, str) else None
    return TableRead(
        table_key=table_key,
        producer_target=producer_str,
        loader_node=node.name,
        loader_type=str(node_type),
    )


def _validate_target_nodes(
    targets: Mapping[str, object],
    target_nodes: Mapping[str, str],
    *,
    strict: bool,
) -> None:
    if not strict:
        return
    missing = sorted(set(targets) - set(target_nodes))
    if missing:
        msg = "Hamilton graph missing materialize nodes for targets: " + ", ".join(missing)
        raise RuntimeError(msg)


__all__ = ["compile_dag_catalog"]
