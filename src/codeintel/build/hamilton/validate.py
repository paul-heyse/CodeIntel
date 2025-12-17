"""Hamilton DAG validator for build invariants.

This module provides a small, deterministic validation gate that checks
Hamilton graph invariants required for a DAG-first architecture.
"""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass
from textwrap import dedent
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Literal, Protocol, cast

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.schemas import get_schema_provider
from codeintel.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_COMPUTE,
    NODE_TYPE_DATASET,
    NODE_TYPE_MATERIALIZE,
    TAG_ARTIFACT,
    TAG_DOMAIN,
    TAG_NODE_TYPE,
    TAG_TABLE_KEY,
    TAG_TARGET,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
        "errors": [_issue_to_obj(i) for i in result.errors],
        "warnings": [_issue_to_obj(i) for i in result.warnings],
        "summary": {
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
        },
    }
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"


def validate_graph() -> GraphValidationResult:
    """Validate the Hamilton graph for build invariants.

    Returns
    -------
    GraphValidationResult
        Validation result for the constructed graph.
    """
    runtime = build_driver()
    return validate_nodes(
        runtime.dr.graph.nodes,
        base_graph=runtime.graph,
        enforce_compute_io_purity=True,
    )


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


def _collect_produced_artifacts(
    nodes: Mapping[str, NodeLike],
    *,
    node_to_target: Mapping[str, str],
) -> tuple[dict[str, str], list[GraphValidationIssue]]:
    produced_artifact_to_target: dict[str, str] = {}
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
            continue

        producer_targets = {
            node_to_target[dep.name] for dep in node.dependencies if dep.name in node_to_target
        }
        if not producer_targets:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_producer",
                    message="Artifact node missing a producing materialize dependency",
                    node=node.name,
                    artifact=artifact,
                )
            )
            continue
        if len(producer_targets) > 1:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="multiple_producers",
                    message=(
                        "Artifact node has multiple producing materialize dependencies: "
                        + ", ".join(sorted(producer_targets))
                    ),
                    node=node.name,
                    artifact=artifact,
                )
            )
            continue

        producer_target = next(iter(producer_targets))
        existing = produced_artifact_to_target.get(artifact)
        if existing is not None and existing != producer_target:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="duplicate_artifact_key",
                    message=f"artifact produced by multiple targets: {existing}, {producer_target}",
                    node=node.name,
                    artifact=artifact,
                    target=producer_target,
                )
            )
        else:
            produced_artifact_to_target[artifact] = producer_target

    return produced_artifact_to_target, issues


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


def _derived_outputs_mismatch_issues(
    *,
    base_graph: TargetGraph,
    produced_table_to_target: Mapping[str, str],
    produced_artifact_to_target: Mapping[str, str],
) -> list[GraphValidationIssue]:
    tables_by_target: dict[str, set[str]] = {}
    for table_key, producer in produced_table_to_target.items():
        tables_by_target.setdefault(producer, set()).add(table_key)

    artifacts_by_target: dict[str, set[str]] = {}
    for artifact, producer in produced_artifact_to_target.items():
        artifacts_by_target.setdefault(producer, set()).add(artifact)

    issues: list[GraphValidationIssue] = []
    for target in sorted(base_graph.all_targets, key=lambda t: t.name):
        derived_tables = tables_by_target.get(target.name, set())
        derived_artifacts = artifacts_by_target.get(target.name, set())

        contract_tables = set(target.contract.table_keys)
        contract_artifacts = set(target.contract.artifact_names)

        if derived_tables != contract_tables:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="contract_tables_mismatch",
                    message=(
                        f"Contract tables do not match DAG-derived tables "
                        f"(contract={sorted(contract_tables)}, derived={sorted(derived_tables)})"
                    ),
                    target=target.name,
                )
            )

        if derived_artifacts != contract_artifacts:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="contract_artifacts_mismatch",
                    message=(
                        f"Contract artifacts do not match DAG-derived artifacts "
                        f"(contract={sorted(contract_artifacts)}, derived={sorted(derived_artifacts)})"
                    ),
                    target=target.name,
                )
            )

    return issues


def _target_tag_value(tags: Mapping[str, object]) -> str | None:
    target = tags.get(TAG_TARGET)
    return target if isinstance(target, str) and target else None


def _compute_node_origin_fn(node: NodeLike) -> FunctionType | MethodType | None:
    originating = getattr(node, "originating_functions", None)
    if not isinstance(originating, (list, tuple)) or not originating:
        return None

    fn = originating[0]
    if not callable(fn):
        return None

    unwrapped = inspect.unwrap(fn)
    if isinstance(unwrapped, (FunctionType, MethodType)):
        return unwrapped
    return None


def _forbidden_calls_in_source(source: str) -> frozenset[str] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    forbidden: set[str] = set()
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        fn_node = call.func
        if not isinstance(fn_node, ast.Attribute):
            continue

        if fn_node.attr in {"execute", "execute_async", "execute_scalar"}:
            forbidden.add(f".{fn_node.attr}()")

        if (
            fn_node.attr == "table"
            and isinstance(fn_node.value, ast.Attribute)
            and fn_node.value.attr == "ibis"
        ):
            forbidden.add(".ibis.table()")

    return frozenset(forbidden)


def _compute_io_purity_issue(
    *,
    node: NodeLike,
    tags: Mapping[str, object],
) -> GraphValidationIssue | None:
    fn = _compute_node_origin_fn(node)
    if fn is None:
        return None

    target = _target_tag_value(tags)

    try:
        raw_source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        return GraphValidationIssue(
            severity="warning",
            code="compute_source_unavailable",
            message=f"Unable to inspect compute node source: {exc}",
            node=node.name,
            target=target,
        )

    forbidden = _forbidden_calls_in_source(dedent(raw_source))
    if forbidden is None:
        return GraphValidationIssue(
            severity="warning",
            code="compute_source_unparseable",
            message="Unable to parse compute node source",
            node=node.name,
            target=target,
        )

    if not forbidden:
        return None

    return GraphValidationIssue(
        severity="error",
        code="compute_io_forbidden",
        message="Compute node contains forbidden IO calls: " + ", ".join(sorted(forbidden)),
        node=node.name,
        target=target,
    )


def _compute_io_purity_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue
        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_COMPUTE:
            continue
        if getattr(node, "user_defined", False):
            continue
        if node.name.endswith(("_raw", "_validator")):
            continue

        issue = _compute_io_purity_issue(node=node, tags=tags)
        if issue is not None:
            issues.append(issue)

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
    base_graph: TargetGraph | None = None,
    schema_provider: SchemaProvider | None = None,
    enforce_compute_io_purity: bool = False,
) -> GraphValidationResult:
    """Validate Hamilton FunctionGraph nodes against build invariants.

    Parameters
    ----------
    nodes
        Mapping of Hamilton node name to node-like objects.
    base_graph
        Optional TargetGraph used for warn-only dependency parity checks.
    schema_provider
        Optional schema provider override (defaults to canonical provider).
    enforce_compute_io_purity
        When True, validate that nodes tagged ``node_type="compute"`` do not contain direct I/O
        calls (e.g., ``.execute()``, ``.execute_scalar()``, or ``.ibis.table()``).

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
    produced_artifact_to_target, artifact_issues = _collect_produced_artifacts(
        nodes, node_to_target=node_to_target
    )

    errors: list[GraphValidationIssue] = [
        *materialize_issues,
        *dataset_issues,
        *artifact_issues,
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
        errors.extend(
            _derived_outputs_mismatch_issues(
                base_graph=base_graph,
                produced_table_to_target=produced_table_to_target,
                produced_artifact_to_target=produced_artifact_to_target,
            )
        )

    if enforce_compute_io_purity:
        io_issues = _compute_io_purity_issues(nodes)
        errors.extend(i for i in io_issues if i.severity == "error")
        warnings.extend(i for i in io_issues if i.severity == "warning")

    errors_sorted = tuple(sorted(errors, key=lambda i: (i.code, i.message, i.node or "")))
    warnings_sorted = tuple(sorted(warnings, key=lambda i: (i.code, i.message, i.node or "")))
    return GraphValidationResult(errors=errors_sorted, warnings=warnings_sorted)


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
