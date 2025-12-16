"""Hamilton DAG validator for build invariants.

This module provides a small, deterministic validation gate that checks
Hamilton graph invariants required for a DAG-first architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_DATASET,
    NODE_TYPE_MATERIALIZE,
    TAG_ARTIFACT,
    TAG_DOMAIN,
    TAG_NODE_TYPE,
    TAG_TABLE_KEY,
    TAG_TARGET,
)
from codeintel.build.schemas import get_schema_provider

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.hamilton.driver_factory import HamiltonNodeMode
    from codeintel.build.targets import TargetGraph
    from codeintel.core.schemas.provider import SchemaProvider


class NodeLike(Protocol):
    """Minimal protocol for Hamilton nodes used by the validator."""

    @property
    def name(self) -> str:
        """Stable node name."""
        ...

    @property
    def tags(self) -> object:
        """Optional node tags payload."""
        ...

    @property
    def dependencies(self) -> Sequence[NodeLike]:
        """Direct upstream node dependencies."""
        ...


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class GraphValidationIssue:
    """Single validation issue produced by the graph validator."""

    severity: Severity
    code: str
    message: str
    node: str | None = None
    target: str | None = None
    table_key: str | None = None
    artifact: str | None = None


@dataclass(frozen=True)
class GraphValidationResult:
    """Validation result for a Hamilton graph."""

    mode: HamiltonNodeMode
    errors: tuple[GraphValidationIssue, ...]
    warnings: tuple[GraphValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        """Return True when the result contains errors."""
        return bool(self.errors)


def _issue_to_obj(issue: GraphValidationIssue) -> dict[str, object]:
    obj: dict[str, object] = {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
    }
    if issue.node is not None:
        obj["node"] = issue.node
    if issue.target is not None:
        obj["target"] = issue.target
    if issue.table_key is not None:
        obj["table_key"] = issue.table_key
    if issue.artifact is not None:
        obj["artifact"] = issue.artifact
    return obj


def validation_result_to_json(
    result: GraphValidationResult,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize a graph validation result to deterministic JSON text.

    Parameters
    ----------
    result
        Graph validation result.
    indent
        JSON indentation level. When None, emits compact JSON.

    Returns
    -------
    str
        Newline-terminated JSON payload.
    """
    obj: dict[str, object] = {
        "mode": result.mode,
        "errors": [_issue_to_obj(i) for i in result.errors],
        "warnings": [_issue_to_obj(i) for i in result.warnings],
        "summary": {
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
        },
    }
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"


def validate_graph(*, mode: HamiltonNodeMode = "auto") -> GraphValidationResult:
    """Validate the Hamilton graph for the selected mode.

    Parameters
    ----------
    mode
        Node generation mode.

    Returns
    -------
    GraphValidationResult
        Validation result for the constructed graph.
    """
    runtime = build_driver(mode=mode)
    return validate_nodes(runtime.dr.graph.nodes, mode=mode, base_graph=runtime.graph)


def _tags_mapping(node: NodeLike) -> Mapping[str, object] | None:
    tags = node.tags
    if not isinstance(tags, dict):
        return None
    return cast("Mapping[str, object]", tags)


def _collect_materialize_index(
    nodes: Mapping[str, NodeLike],
) -> tuple[dict[str, str], dict[str, list[str]], list[GraphValidationIssue]]:
    node_to_target: dict[str, str] = {}
    materialize_nodes_by_target: dict[str, list[str]] = {}
    issues: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue

        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_MATERIALIZE:
            continue

        domain = tags.get(TAG_DOMAIN)
        target = tags.get(TAG_TARGET)
        if not isinstance(domain, str) or not domain:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Materialize node missing domain tag",
                    node=node.name,
                )
            )
        if not isinstance(target, str) or not target:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Materialize node missing target tag",
                    node=node.name,
                )
            )
            continue

        node_to_target[node.name] = target
        materialize_nodes_by_target.setdefault(target, []).append(node.name)

    return node_to_target, materialize_nodes_by_target, issues


def _collect_produced_tables(
    nodes: Mapping[str, NodeLike],
    *,
    node_to_target: Mapping[str, str],
) -> tuple[dict[str, str], list[GraphValidationIssue]]:
    produced_table_to_target: dict[str, str] = {}
    issues: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue

        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_DATASET:
            continue

        domain = tags.get(TAG_DOMAIN)
        table_key = tags.get(TAG_TABLE_KEY)
        if not isinstance(domain, str) or not domain:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Dataset node missing domain tag",
                    node=node.name,
                )
            )
        if not isinstance(table_key, str) or not table_key:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Dataset node missing table_key tag",
                    node=node.name,
                )
            )
            continue

        producer_targets = {
            node_to_target[dep.name] for dep in node.dependencies if dep.name in node_to_target
        }
        if not producer_targets:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_producer",
                    message="Dataset node missing a producing materialize dependency",
                    node=node.name,
                    table_key=table_key,
                )
            )
            continue
        if len(producer_targets) > 1:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="multiple_producers",
                    message=(
                        "Dataset node has multiple producing materialize dependencies: "
                        + ", ".join(sorted(producer_targets))
                    ),
                    node=node.name,
                    table_key=table_key,
                )
            )
            continue

        producer_target = next(iter(producer_targets))
        existing = produced_table_to_target.get(table_key)
        if existing is not None and existing != producer_target:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="duplicate_table_key",
                    message=f"table_key produced by multiple targets: {existing}, {producer_target}",
                    node=node.name,
                    table_key=table_key,
                    target=producer_target,
                )
            )
        else:
            produced_table_to_target[table_key] = producer_target

    return produced_table_to_target, issues


def _artifact_tag_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue

        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_ARTIFACT:
            continue

        domain = tags.get(TAG_DOMAIN)
        artifact = tags.get(TAG_ARTIFACT)
        if not isinstance(domain, str) or not domain:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Artifact node missing domain tag",
                    node=node.name,
                )
            )
        if not isinstance(artifact, str) or not artifact:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Artifact node missing artifact tag",
                    node=node.name,
                )
            )

    return issues


def _duplicate_materialize_issues(
    materialize_nodes_by_target: Mapping[str, Sequence[str]],
) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    for target, node_names in sorted(materialize_nodes_by_target.items()):
        if len(node_names) > 1:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="duplicate_materialize",
                    message=f"Multiple materialize nodes declared for target: {', '.join(node_names)}",
                    target=target,
                )
            )
    return issues


def _unknown_schema_issues(
    *,
    provider: SchemaProvider,
    produced_table_to_target: Mapping[str, str],
) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    for table_key, producer_target in sorted(produced_table_to_target.items()):
        try:
            provider.require_table_schema(table_key)
        except (KeyError, RuntimeError, ValueError) as exc:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="unknown_table_schema",
                    message=str(exc),
                    table_key=table_key,
                    target=producer_target,
                )
            )
    return issues


def _deps_mismatch_warnings(
    *,
    derived_deps: Mapping[str, Sequence[str]],
    base_graph: TargetGraph,
) -> list[GraphValidationIssue]:
    warnings: list[GraphValidationIssue] = []
    for target in sorted(base_graph.all_targets, key=lambda t: t.name):
        derived = set(derived_deps.get(target.name, ()))
        declared = set(target.dependencies)
        if derived != declared:
            warnings.append(
                GraphValidationIssue(
                    severity="warning",
                    code="deps_mismatch",
                    message=(
                        f"Declared deps drift from Hamilton-derived deps "
                        f"(declared={sorted(declared)}, derived={sorted(derived)})"
                    ),
                    target=target.name,
                )
            )
    return warnings


def validate_nodes(
    nodes: Mapping[str, NodeLike],
    *,
    mode: HamiltonNodeMode,
    base_graph: TargetGraph | None = None,
    schema_provider: SchemaProvider | None = None,
) -> GraphValidationResult:
    """Validate Hamilton FunctionGraph nodes against build invariants.

    Parameters
    ----------
    nodes
        Mapping of Hamilton node name to node-like objects.
    mode
        Node mode used to construct the graph.
    base_graph
        Optional TargetGraph used for warn-only dependency parity checks.
    schema_provider
        Optional schema provider override (defaults to canonical provider).

    Returns
    -------
    GraphValidationResult
        Validation results with deterministic ordering.
    """
    provider = get_schema_provider() if schema_provider is None else schema_provider

    node_to_target, materialize_nodes_by_target, materialize_issues = _collect_materialize_index(
        nodes
    )
    produced_table_to_target, dataset_issues = _collect_produced_tables(
        nodes, node_to_target=node_to_target
    )

    errors: list[GraphValidationIssue] = [
        *materialize_issues,
        *dataset_issues,
        *_artifact_tag_issues(nodes),
    ]
    errors.extend(_duplicate_materialize_issues(materialize_nodes_by_target))
    errors.extend(
        _unknown_schema_issues(provider=provider, produced_table_to_target=produced_table_to_target)
    )

    warnings: list[GraphValidationIssue] = []

    derived_deps: dict[str, tuple[str, ...]] | None = None
    if not any(issue.code == "duplicate_materialize" for issue in errors):
        derived_deps = _derive_target_dependencies(nodes, node_to_target=node_to_target)
        cycles = _find_cycles(derived_deps)
        errors.extend(
            GraphValidationIssue(
                severity="error",
                code="cycle_detected",
                message="Target dependency cycle detected: " + " -> ".join(cycle),
            )
            for cycle in cycles
        )

    if derived_deps is not None and base_graph is not None:
        warnings.extend(_deps_mismatch_warnings(derived_deps=derived_deps, base_graph=base_graph))

    errors_sorted = tuple(sorted(errors, key=lambda i: (i.code, i.message, i.node or "")))
    warnings_sorted = tuple(sorted(warnings, key=lambda i: (i.code, i.message, i.node or "")))
    return GraphValidationResult(mode=mode, errors=errors_sorted, warnings=warnings_sorted)


def _derive_target_dependencies(
    nodes: Mapping[str, NodeLike],
    *,
    node_to_target: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    target_to_node_name: dict[str, str] = {}
    for node_name, target in node_to_target.items():
        if target in target_to_node_name:
            continue
        target_to_node_name[target] = node_name

    derived: dict[str, tuple[str, ...]] = {}
    for target, node_name in sorted(target_to_node_name.items()):
        root = nodes[node_name]
        deps = _direct_target_dependencies(
            root=root, root_target=target, node_to_target=node_to_target
        )
        derived[target] = tuple(sorted(deps))

    return derived


def _direct_target_dependencies(
    *,
    root: NodeLike,
    root_target: str,
    node_to_target: Mapping[str, str],
) -> frozenset[str]:
    deps: set[str] = set()
    visited: set[str] = set()
    stack: list[NodeLike] = list(root.dependencies)

    while stack:
        node = stack.pop()
        if node.name in visited:
            continue
        visited.add(node.name)

        target = node_to_target.get(node.name)
        if target is not None:
            if target != root_target:
                deps.add(target)
            continue

        stack.extend(node.dependencies)

    return frozenset(deps)


def _find_cycles(graph: Mapping[str, Sequence[str]]) -> list[tuple[str, ...]]:
    visited: set[str] = set()
    stack: list[str] = []
    stack_index: dict[str, int] = {}
    cycles: list[tuple[str, ...]] = []
    cycle_keys: set[tuple[str, ...]] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        stack_index[node] = len(stack)
        stack.append(node)

        for dep in sorted(graph.get(node, ())):
            if dep not in visited:
                dfs(dep)
                continue
            if dep in stack_index:
                start = stack_index[dep]
                cycle = (*stack[start:], dep)
                if cycle not in cycle_keys:
                    cycles.append(cycle)
                    cycle_keys.add(cycle)

        stack.pop()
        stack_index.pop(node, None)

    for node in sorted(graph):
        if node not in visited:
            dfs(node)

    return cycles


__all__ = [
    "GraphValidationIssue",
    "GraphValidationResult",
    "NodeLike",
    "validate_graph",
    "validate_nodes",
    "validation_result_to_json",
]
