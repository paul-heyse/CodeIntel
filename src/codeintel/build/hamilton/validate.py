"""Hamilton DAG validator for build invariants.

This module provides a small, deterministic validation gate that checks
Hamilton graph invariants required for a DAG-first architecture.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections import Counter
from dataclasses import dataclass, replace
from textwrap import dedent
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Literal, Protocol, cast

from duckdb import Error as DuckDBError
from polars import exceptions as polars_exceptions

from codeintel.build.hamilton.naming import target_node
from codeintel.core.hamilton.semantic_tags import (
    TAG_SEMANTIC_ENTITY,
    TAG_SEMANTIC_GRAIN,
    TAG_SEMANTIC_KIND,
)
from codeintel.core.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_COMPUTE,
    NODE_TYPE_DATASET,
    NODE_TYPE_HELPER,
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_MATERIALIZE,
    NODE_TYPE_TOOL,
    OUTPUT_KIND_SEMANTIC_VIEW,
    TAG_ARTIFACT,
    TAG_ARTIFACT_PATH_TEMPLATE,
    TAG_DOMAIN,
    TAG_ENTITY,
    TAG_ENTITY_KEYS,
    TAG_GRAIN,
    TAG_JOIN_KEYS,
    TAG_KIND,
    TAG_LAYER,
    TAG_MCP_VISIBLE,
    TAG_NODE_TYPE,
    TAG_OUTPUT_KIND,
    TAG_SCHEMA_REF,
    TAG_SEMANTIC_ID,
    TAG_TABLE_KEY,
    TAG_TARGET,
    TAG_TARGET_SPEC_VERSION,
    TAG_VERSION,
)
from codeintel.core.imports.lazy import lazy_getattr

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from codeintel.core.schemas.provider import SchemaProvider

    class ModuleProvenanceLike(Protocol):
        @property
        def origin(self) -> str: ...

        @property
        def module_import(self) -> str: ...

        @property
        def plugin_name(self) -> str | None: ...

        @property
        def dist_name(self) -> str | None: ...

        @property
        def dist_version(self) -> str | None: ...


_SCHEMA_PROVIDER_FACTORY = cast(
    "Callable[[], SchemaProvider]",
    lazy_getattr("codeintel.build.schemas.registry", "get_schema_provider"),
)


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
    module: str | None = None
    origin: str | None = None
    module_import: str | None = None
    plugin_name: str | None = None
    dist_name: str | None = None
    dist_version: str | None = None


@dataclass(frozen=True)
class GraphValidationResult:
    """Validation result for a Hamilton graph."""

    errors: tuple[GraphValidationIssue, ...]
    warnings: tuple[GraphValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        """Return True when the result contains errors."""
        return bool(self.errors)


@dataclass(frozen=True, slots=True)
class _ValidationInputs:
    node_to_target: Mapping[str, str]
    materialize_nodes_by_target: Mapping[str, list[str]]
    materialize_issues: list[GraphValidationIssue]
    dataset_issues: list[GraphValidationIssue]
    artifact_issues: list[GraphValidationIssue]
    saver_table_to_target: dict[str, str]
    saver_artifact_to_target: dict[str, str]
    saver_issues: list[GraphValidationIssue]


def _issue_to_obj(
    issue: GraphValidationIssue,
    *,
    node_provenance: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    obj: dict[str, object] = {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
    }
    optional_fields = {
        "node": issue.node,
        "target": issue.target,
        "table_key": issue.table_key,
        "artifact": issue.artifact,
        "module": issue.module,
        "origin": issue.origin,
        "module_import": issue.module_import,
        "plugin_name": issue.plugin_name,
        "dist_name": issue.dist_name,
        "dist_version": issue.dist_version,
    }
    obj.update({key: value for key, value in optional_fields.items() if value is not None})
    if issue.node is not None and node_provenance is not None:
        provenance = node_provenance.get(issue.node)
        if provenance:
            obj["node_provenance"] = dict(provenance)
    return obj


def validation_result_to_json(
    result: GraphValidationResult,
    *,
    indent: int | None = 2,
    node_provenance: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Serialize a graph validation result to deterministic JSON text.

    Parameters
    ----------
    result
        Graph validation result.
    indent
        JSON indentation level. When None, emits compact JSON.
    node_provenance
        Optional mapping of node name to provenance metadata.

    Returns
    -------
    str
        Newline-terminated JSON payload.
    """
    obj: dict[str, object] = {
        "errors": [_issue_to_obj(i, node_provenance=node_provenance) for i in result.errors],
        "warnings": [_issue_to_obj(i, node_provenance=node_provenance) for i in result.warnings],
        "summary": {
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
        },
    }
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"


def _tags_mapping(node: NodeLike) -> Mapping[str, object] | None:
    tags = node.tags
    if not isinstance(tags, dict):
        return None
    return cast("Mapping[str, object]", tags)


_ALLOWED_NODE_TYPES: frozenset[str] = frozenset(
    {
        NODE_TYPE_ARTIFACT,
        NODE_TYPE_COMPUTE,
        NODE_TYPE_DATASET,
        NODE_TYPE_HELPER,
        NODE_TYPE_LOADER_QUERY,
        NODE_TYPE_MATERIALIZE,
        NODE_TYPE_TOOL,
    }
)

_ALLOWED_OUTPUT_ROLES: frozenset[str] = frozenset({"contract", "internal"})
_ALLOWED_MCP_VISIBLE: frozenset[str] = frozenset({"0", "1"})


def _tag_type_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue

        node_type = tags.get(TAG_NODE_TYPE)
        if node_type is not None and (
            not isinstance(node_type, str) or node_type not in _ALLOWED_NODE_TYPES
        ):
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="invalid_tag",
                    message=f"Invalid node_type tag: {node_type!r}",
                    node=node.name,
                )
            )

        mcp_visible = tags.get(TAG_MCP_VISIBLE)
        if mcp_visible is not None and (
            not isinstance(mcp_visible, str) or mcp_visible not in _ALLOWED_MCP_VISIBLE
        ):
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="invalid_tag",
                    message=f"Invalid mcp_visible tag: {mcp_visible!r}",
                    node=node.name,
                )
            )

        data_saver = tags.get("hamilton.data_saver")
        if data_saver is not None and not isinstance(data_saver, bool):
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="invalid_tag",
                    message=f"Invalid hamilton.data_saver tag: {data_saver!r}",
                    node=node.name,
                )
            )

        output_role = tags.get("output_role")
        if output_role is not None and output_role not in _ALLOWED_OUTPUT_ROLES:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="invalid_tag",
                    message=f"Invalid output_role tag: {output_role!r}",
                    node=node.name,
                )
            )

        for key, value in tags.items():
            if isinstance(value, list) and not all(isinstance(item, str) for item in value):
                issues.append(
                    GraphValidationIssue(
                        severity="error",
                        code="invalid_tag",
                        message=f"Invalid list tag {key!r}: {value!r}",
                        node=node.name,
                    )
                )

    return issues


def _cpg2_naming_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    for node in nodes.values():
        module_name = _node_module_name(node)
        if module_name is None:
            continue
        if ".graphs.cpg2" not in module_name:
            continue
        if node.name.startswith("cpg2_"):
            continue
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="cpg2_unprefixed_node",
                message="CPG2 node must be prefixed with 'cpg2_'.",
                node=node.name,
                module=module_name,
            )
        )
    return issues


def _cpg_aggregator_dependency_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    nodes_node = nodes.get("cpg_nodes")
    if nodes_node is not None:
        deps = {dep.name for dep in nodes_node.dependencies}
        expected = {"cpg2_nodes__frames"}
        if deps != expected:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="cpg_nodes_dependencies",
                    message=(
                        "cpg_nodes must depend only on cpg2_nodes__frames "
                        f"(found {sorted(deps)})."
                    ),
                    node=nodes_node.name,
                )
            )
    edges_node = nodes.get("cpg_edges")
    if edges_node is not None:
        deps = {dep.name for dep in edges_node.dependencies}
        expected = {"cpg2_edges__frames"}
        if deps != expected:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="cpg_edges_dependencies",
                    message=(
                        "cpg_edges must depend only on cpg2_edges__frames "
                        f"(found {sorted(deps)})."
                    ),
                    node=edges_node.name,
                )
            )
    return issues


def _duplicate_node_name_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    counts = Counter(node.name for node in nodes.values())
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    for name in duplicates:
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="duplicate_node_name",
                message=f"Duplicate node name discovered: {name}",
                node=name,
            )
        )
    return issues


def _target_anchor_tag_issues(
    nodes: Mapping[str, NodeLike],
) -> tuple[list[GraphValidationIssue], list[GraphValidationIssue]]:
    errors: list[GraphValidationIssue] = []
    warnings: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue
        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_MATERIALIZE:
            continue
        if not node_name.startswith("t__"):
            continue

        target = _tag_str_value(tags, TAG_TARGET)
        if target is not None:
            expected_name = target_node(target)
            if node_name != expected_name:
                errors.append(
                    GraphValidationIssue(
                        severity="error",
                        code="invalid_anchor_name",
                        message=(
                            "Target anchor name does not match target tag "
                            f"(expected {expected_name})"
                        ),
                        node=node_name,
                        target=target,
                    )
                )

        kind = _tag_str_value(tags, TAG_KIND)
        if kind is None:
            warnings.append(
                GraphValidationIssue(
                    severity="warning",
                    code="missing_anchor_tag",
                    message="Target anchor missing kind tag",
                    node=node_name,
                    target=target,
                )
            )
        schema_ref = _tag_str_value(tags, TAG_SCHEMA_REF)
        if schema_ref is None:
            warnings.append(
                GraphValidationIssue(
                    severity="warning",
                    code="missing_anchor_tag",
                    message="Target anchor missing schema_ref tag",
                    node=node_name,
                    target=target,
                )
            )

    return errors, warnings


def _tag_str_value(tags: Mapping[str, object], key: str) -> str | None:
    value = tags.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _semantic_tag_value(
    tags: Mapping[str, object],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = _tag_str_value(tags, key)
        if value is not None:
            return value
    return None


def _semantic_kind(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (TAG_KIND, TAG_SEMANTIC_KIND))


def _semantic_entity(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (TAG_ENTITY, TAG_SEMANTIC_ENTITY))


def _semantic_grain(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (TAG_GRAIN, TAG_SEMANTIC_GRAIN))


def _is_semantic_output(tags: Mapping[str, object]) -> bool:
    output_kind = _tag_str_value(tags, TAG_OUTPUT_KIND)
    layer = _tag_str_value(tags, TAG_LAYER)
    return output_kind == OUTPUT_KIND_SEMANTIC_VIEW or layer == "semantic"


def _missing_semantic_tag_issue(
    *,
    node: NodeLike,
    target: str | None,
    message: str,
    code: str = "missing_semantic_tag",
) -> GraphValidationIssue:
    return GraphValidationIssue(
        severity="error",
        code=code,
        message=message,
        node=node.name,
        target=target,
    )


def _semantic_layer_issue(
    *,
    node: NodeLike,
    tags: Mapping[str, object],
    target: str | None,
) -> GraphValidationIssue | None:
    layer = _tag_str_value(tags, TAG_LAYER)
    if layer == "semantic":
        return None
    code = "missing_semantic_tag" if layer is None else "invalid_semantic_tag"
    detail = "missing layer tag" if layer is None else f"layer tag must be 'semantic' (got {layer})"
    return _missing_semantic_tag_issue(
        node=node,
        target=target,
        message=f"Semantic output {detail}",
        code=code,
    )


def _semantic_required_tag_issues(
    *,
    node: NodeLike,
    tags: Mapping[str, object],
    target: str | None,
    kind: str | None,
) -> list[GraphValidationIssue]:
    required = (
        (_tag_str_value(tags, TAG_SEMANTIC_ID), "Semantic output missing semantic_id tag"),
        (kind, "Semantic output missing kind tag"),
        (_semantic_entity(tags), "Semantic output missing entity tag"),
        (_semantic_grain(tags), "Semantic output missing grain tag"),
        (_tag_str_value(tags, TAG_VERSION), "Semantic output missing version tag"),
    )
    issues: list[GraphValidationIssue] = []
    for value, message in required:
        if value is None:
            issues.append(_missing_semantic_tag_issue(node=node, target=target, message=message))
    return issues


def _semantic_table_tag_issues(
    *,
    node: NodeLike,
    tags: Mapping[str, object],
    target: str | None,
    kind: str | None,
) -> list[GraphValidationIssue]:
    if kind != "table":
        return []
    required = (
        (TAG_SCHEMA_REF, "Semantic table missing schema_ref tag"),
        (TAG_ENTITY_KEYS, "Semantic table missing entity_keys tag"),
        (TAG_JOIN_KEYS, "Semantic table missing join_keys tag"),
    )
    issues: list[GraphValidationIssue] = []
    for key, message in required:
        if _tag_str_value(tags, key) is None:
            issues.append(_missing_semantic_tag_issue(node=node, target=target, message=message))
    return issues


def _semantic_tag_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue
        if not _is_semantic_output(tags):
            continue

        target = _target_tag_value(tags)
        layer_issue = _semantic_layer_issue(node=node, tags=tags, target=target)
        if layer_issue is not None:
            issues.append(layer_issue)
        kind = _semantic_kind(tags)
        issues.extend(
            _semantic_required_tag_issues(
                node=node,
                tags=tags,
                target=target,
                kind=kind,
            )
        )
        issues.extend(
            _semantic_table_tag_issues(
                node=node,
                tags=tags,
                target=target,
                kind=kind,
            )
        )

    return issues


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

        spec_version = tags.get(TAG_TARGET_SPEC_VERSION)
        if spec_version != "1":
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Materialize node missing/invalid target_spec_version tag",
                    node=node.name,
                    target=target,
                )
            )

        node_to_target[node.name] = target
        materialize_nodes_by_target.setdefault(target, []).append(node.name)

    return node_to_target, materialize_nodes_by_target, issues


def _require_tag_value(
    *,
    tags: Mapping[str, object],
    key: str,
    message: str,
    node_name: str,
    issues: list[GraphValidationIssue],
) -> str | None:
    value = tags.get(key)
    if isinstance(value, str) and value:
        return value
    issues.append(
        GraphValidationIssue(
            severity="error",
            code="missing_tag",
            message=message,
            node=node_name,
        )
    )
    return None


def _candidate_targets(
    tags: Mapping[str, object],
    available_targets: tuple[str, ...],
) -> tuple[str, ...]:
    target = _target_tag_value(tags)
    return available_targets if target is None else (target,)


@dataclass(frozen=True, slots=True)
class _ProducerContext:
    node_to_target: Mapping[str, str]
    nodes: Mapping[str, NodeLike]
    dependents: Mapping[str, Sequence[str]]
    available_targets: tuple[str, ...]


def _resolve_producer_targets(
    node: NodeLike,
    *,
    tags: Mapping[str, object],
    context: _ProducerContext,
) -> set[str]:
    producer_targets = {
        context.node_to_target[dep.name]
        for dep in node.dependencies
        if dep.name in context.node_to_target
    }
    if producer_targets:
        return producer_targets
    candidate_targets = _candidate_targets(tags, context.available_targets)
    return {
        candidate
        for candidate in candidate_targets
        if _materialize_node_reachable(
            start_node=node.name,
            target=candidate,
            nodes=context.nodes,
            dependents=context.dependents,
        )
    }


def _record_produced_table(
    *,
    produced: dict[str, str],
    table_key: str,
    target: str,
    node_name: str,
    issues: list[GraphValidationIssue],
) -> None:
    existing = produced.get(table_key)
    if existing is not None and existing != target:
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="duplicate_table_key",
                message=f"table_key produced by multiple targets: {existing}, {target}",
                node=node_name,
                table_key=table_key,
                target=target,
            )
        )
        return
    produced[table_key] = target


def _record_produced_artifact(
    *,
    produced: dict[str, str],
    artifact: str,
    target: str,
    node_name: str,
    issues: list[GraphValidationIssue],
) -> None:
    existing = produced.get(artifact)
    if existing is not None and existing != target:
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="duplicate_artifact_key",
                message=f"artifact produced by multiple targets: {existing}, {target}",
                node=node_name,
                artifact=artifact,
                target=target,
            )
        )
        return
    produced[artifact] = target


def _collect_produced_tables(
    nodes: Mapping[str, NodeLike],
    *,
    node_to_target: Mapping[str, str],
    materialize_nodes_by_target: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str], list[GraphValidationIssue]]:
    produced_table_to_target: dict[str, str] = {}
    issues: list[GraphValidationIssue] = []
    dependents = _build_dependents_map(nodes)
    available_targets = tuple(materialize_nodes_by_target.keys())
    producer_context = _ProducerContext(
        node_to_target=node_to_target,
        nodes=nodes,
        dependents=dependents,
        available_targets=available_targets,
    )

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue

        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_DATASET:
            continue

        _require_tag_value(
            tags=tags,
            key=TAG_DOMAIN,
            message="Dataset node missing domain tag",
            node_name=node.name,
            issues=issues,
        )
        table_key = _require_tag_value(
            tags=tags,
            key=TAG_TABLE_KEY,
            message="Dataset node missing table_key tag",
            node_name=node.name,
            issues=issues,
        )
        if table_key is None:
            continue

        producer_targets = _resolve_producer_targets(
            node,
            tags=tags,
            context=producer_context,
        )
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
        _record_produced_table(
            produced=produced_table_to_target,
            table_key=table_key,
            target=producer_target,
            node_name=node.name,
            issues=issues,
        )

    return produced_table_to_target, issues


def _collect_produced_artifacts(
    nodes: Mapping[str, NodeLike],
    *,
    node_to_target: Mapping[str, str],
    materialize_nodes_by_target: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str], list[GraphValidationIssue]]:
    produced_artifact_to_target: dict[str, str] = {}
    issues: list[GraphValidationIssue] = []
    dependents = _build_dependents_map(nodes)
    available_targets = tuple(materialize_nodes_by_target.keys())
    producer_context = _ProducerContext(
        node_to_target=node_to_target,
        nodes=nodes,
        dependents=dependents,
        available_targets=available_targets,
    )

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue

        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_ARTIFACT:
            continue

        _require_tag_value(
            tags=tags,
            key=TAG_DOMAIN,
            message="Artifact node missing domain tag",
            node_name=node.name,
            issues=issues,
        )
        artifact = _require_tag_value(
            tags=tags,
            key=TAG_ARTIFACT,
            message="Artifact node missing artifact tag",
            node_name=node.name,
            issues=issues,
        )
        if artifact is None:
            continue

        producer_targets = _resolve_producer_targets(
            node,
            tags=tags,
            context=producer_context,
        )
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
        _record_produced_artifact(
            produced=produced_artifact_to_target,
            artifact=artifact,
            target=producer_target,
            node_name=node.name,
            issues=issues,
        )

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


def _record_saver_table(
    *,
    produced: dict[str, str],
    table_key: str,
    target: str,
    node_name: str,
    issues: list[GraphValidationIssue],
) -> None:
    existing = produced.get(table_key)
    if existing is not None and existing != target:
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="duplicate_table_key",
                message=f"table_key produced by multiple targets: {existing}, {target}",
                node=node_name,
                target=target,
                table_key=table_key,
            )
        )
    produced[table_key] = target


def _record_saver_artifact(
    *,
    produced: dict[str, str],
    artifact: str,
    target: str,
    node_name: str,
    issues: list[GraphValidationIssue],
) -> None:
    existing = produced.get(artifact)
    if existing is not None and existing != target:
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="duplicate_artifact",
                message=f"artifact produced by multiple targets: {existing}, {target}",
                node=node_name,
                target=target,
                artifact=artifact,
            )
        )
    produced[artifact] = target


def _validate_saver_tags(
    *,
    node_name: str,
    tags: Mapping[str, object],
    issues: list[GraphValidationIssue],
) -> tuple[str | None, bool]:
    target = tags.get(TAG_TARGET)
    if not isinstance(target, str) or not target:
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="missing_tag",
                message="DataSaver node missing target tag",
                node=node_name,
            )
        )
        return None, False

    output_role = tags.get("output_role")
    if output_role not in {"contract", "internal"}:
        issues.append(
            GraphValidationIssue(
                severity="error",
                code="missing_tag",
                message="DataSaver node missing/invalid output_role tag",
                node=node_name,
                target=target,
            )
        )
        return target, False

    return target, output_role == "contract"


def _collect_saver_outputs(
    nodes: Mapping[str, NodeLike],
) -> tuple[dict[str, str], dict[str, str], list[GraphValidationIssue]]:
    produced_table_to_target: dict[str, str] = {}
    produced_artifact_to_target: dict[str, str] = {}
    issues: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue

        target, is_contract = _validate_saver_tags(
            node_name=node.name,
            tags=tags,
            issues=issues,
        )
        if target is None or not is_contract:
            continue

        table_key = tags.get(TAG_TABLE_KEY)
        artifact = tags.get(TAG_ARTIFACT)
        table_key_str = table_key if isinstance(table_key, str) and table_key else None
        artifact_str = artifact if isinstance(artifact, str) and artifact else None

        if table_key_str is not None:
            _record_saver_table(
                produced=produced_table_to_target,
                table_key=table_key_str,
                target=target,
                node_name=node.name,
                issues=issues,
            )
        if artifact_str is not None:
            template = tags.get(TAG_ARTIFACT_PATH_TEMPLATE)
            if not isinstance(template, str) or not template:
                issues.append(
                    GraphValidationIssue(
                        severity="error",
                        code="missing_tag",
                        message="Contract DataSaver node missing artifact_path_template tag",
                        node=node.name,
                        target=target,
                        artifact=artifact_str,
                    )
                )
            _record_saver_artifact(
                produced=produced_artifact_to_target,
                artifact=artifact_str,
                target=target,
                node_name=node.name,
                issues=issues,
            )
        if table_key_str is None and artifact_str is None:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="Contract DataSaver node missing table_key/artifact tags",
                    node=node.name,
                    target=target,
                )
            )

    return produced_table_to_target, produced_artifact_to_target, issues


def _collect_validation_inputs(nodes: Mapping[str, NodeLike]) -> _ValidationInputs:
    node_to_target, materialize_nodes_by_target, materialize_issues = _collect_materialize_index(
        nodes
    )
    _, dataset_issues = _collect_produced_tables(
        nodes,
        node_to_target=node_to_target,
        materialize_nodes_by_target=materialize_nodes_by_target,
    )
    _, artifact_issues = _collect_produced_artifacts(
        nodes,
        node_to_target=node_to_target,
        materialize_nodes_by_target=materialize_nodes_by_target,
    )
    saver_table_to_target, saver_artifact_to_target, saver_issues = _collect_saver_outputs(nodes)
    return _ValidationInputs(
        node_to_target=node_to_target,
        materialize_nodes_by_target=materialize_nodes_by_target,
        materialize_issues=materialize_issues,
        dataset_issues=dataset_issues,
        artifact_issues=artifact_issues,
        saver_table_to_target=saver_table_to_target,
        saver_artifact_to_target=saver_artifact_to_target,
        saver_issues=saver_issues,
    )


def _build_dependents_map(nodes: Mapping[str, NodeLike]) -> dict[str, tuple[str, ...]]:
    dependents: dict[str, set[str]] = {}
    for node in nodes.values():
        for dep in node.dependencies:
            dependents.setdefault(dep.name, set()).add(node.name)
    return {name: tuple(sorted(children)) for name, children in dependents.items()}


def _contract_saver_target(node: NodeLike) -> str | None:
    tags = _tags_mapping(node)
    if tags is None:
        return None
    if tags.get("hamilton.data_saver") is not True:
        return None
    if tags.get("output_role") != "contract":
        return None
    return _target_tag_value(tags)


def _materialize_node_reachable(
    *,
    start_node: str,
    target: str,
    nodes: Mapping[str, NodeLike],
    dependents: Mapping[str, Sequence[str]],
) -> bool:
    visited: set[str] = set()
    queue: list[str] = [start_node]

    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        for dependent_name in dependents.get(current, ()):
            dep_node = nodes.get(dependent_name)
            if dep_node is None:
                continue
            dep_tags = _tags_mapping(dep_node)
            if dep_tags is None:
                continue
            if (
                dep_tags.get(TAG_NODE_TYPE) == NODE_TYPE_MATERIALIZE
                and dep_tags.get(TAG_TARGET) == target
            ):
                return True
            queue.append(dependent_name)
    return False


def _orphan_issue_for_saver(
    *,
    node: NodeLike,
    target: str,
    nodes: Mapping[str, NodeLike],
    dependents: Mapping[str, Sequence[str]],
    materialize_nodes_by_target: Mapping[str, Sequence[str]],
) -> GraphValidationIssue | None:
    if target not in materialize_nodes_by_target:
        return GraphValidationIssue(
            severity="error",
            code="orphan_saver",
            message="Contract DataSaver node missing downstream materialize node",
            node=node.name,
            target=target,
        )
    if _materialize_node_reachable(
        start_node=node.name,
        target=target,
        nodes=nodes,
        dependents=dependents,
    ):
        return None
    return GraphValidationIssue(
        severity="error",
        code="orphan_saver",
        message="Contract DataSaver node not connected to target materialize node",
        node=node.name,
        target=target,
    )


def _orphan_saver_issues(
    *,
    nodes: Mapping[str, NodeLike],
    materialize_nodes_by_target: Mapping[str, Sequence[str]],
) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    dependents = _build_dependents_map(nodes)

    for node_name in sorted(nodes):
        node = nodes[node_name]
        target = _contract_saver_target(node)
        if target is None:
            continue
        issue = _orphan_issue_for_saver(
            node=node,
            target=target,
            nodes=nodes,
            dependents=dependents,
            materialize_nodes_by_target=materialize_nodes_by_target,
        )
        if issue is not None:
            issues.append(issue)

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


def _node_module_name(node: NodeLike) -> str | None:
    originating_module = getattr(node, "originating_module", None)
    if isinstance(originating_module, str) and originating_module:
        return originating_module
    fn = _compute_node_origin_fn(node)
    if fn is None:
        return None
    module = getattr(fn, "__module__", None)
    if isinstance(module, str) and module:
        return module
    return None


def _issue_with_provenance(
    issue: GraphValidationIssue,
    *,
    nodes: Mapping[str, NodeLike],
    module_provenance: Mapping[str, ModuleProvenanceLike],
) -> GraphValidationIssue:
    if issue.node is None:
        return issue
    node = nodes.get(issue.node)
    if node is None:
        return issue
    module_name = _node_module_name(node)
    if module_name is None:
        return issue
    provenance = module_provenance.get(module_name)
    if provenance is None:
        return replace(issue, module=module_name)
    return replace(
        issue,
        module=module_name,
        origin=provenance.origin,
        module_import=provenance.module_import,
        plugin_name=provenance.plugin_name,
        dist_name=provenance.dist_name,
        dist_version=provenance.dist_version,
    )


def _attach_issue_provenance(
    issues: list[GraphValidationIssue],
    *,
    nodes: Mapping[str, NodeLike],
    module_provenance: Mapping[str, ModuleProvenanceLike],
) -> list[GraphValidationIssue]:
    return [
        _issue_with_provenance(
            issue,
            nodes=nodes,
            module_provenance=module_provenance,
        )
        for issue in issues
    ]


def _async_node_issue(
    *,
    node: NodeLike,
    tags: Mapping[str, object],
) -> GraphValidationIssue | None:
    fn = _compute_node_origin_fn(node)
    if fn is None:
        return None
    if not inspect.iscoroutinefunction(fn):
        return None
    target = _target_tag_value(tags)
    return GraphValidationIssue(
        severity="error",
        code="async_node_forbidden",
        message="Hamilton nodes must not be async; resolve async work inside the node",
        node=node.name,
        target=target,
    )


def _async_node_issues(nodes: Mapping[str, NodeLike]) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue
        if getattr(node, "user_defined", False):
            continue
        issue = _async_node_issue(node=node, tags=tags)
        if issue is not None:
            issues.append(issue)
    return issues


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
                    message=(
                        f"Multiple materialize nodes declared for target: {', '.join(node_names)}"
                    ),
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
        except (DuckDBError, polars_exceptions.PolarsError) as exc:
            issues.append(
                GraphValidationIssue(
                    severity="warning",
                    code="schema_validation_failed",
                    message=f"Schema validation failed for {table_key}: {exc}",
                    table_key=table_key,
                    target=producer_target,
                )
            )
    return issues


def validate_nodes(
    nodes: Mapping[str, NodeLike],
    *,
    schema_provider: SchemaProvider | None = None,
    validate_schema: bool = True,
    enforce_compute_io_purity: bool = False,
    module_provenance: Mapping[str, ModuleProvenanceLike] | None = None,
) -> GraphValidationResult:
    """Validate Hamilton FunctionGraph nodes against build invariants.

    Parameters
    ----------
    nodes
        Mapping of Hamilton node name to node-like objects.
    schema_provider
        Optional schema provider override (defaults to canonical provider).
    validate_schema
        When False, skip schema provider resolution and unknown schema checks.
    enforce_compute_io_purity
        When True, validate that nodes tagged ``node_type="compute"`` do not contain direct I/O
        calls (e.g., ``.execute()``, ``.execute_scalar()``, or ``.ibis.table()``).
    module_provenance
        Optional mapping of module name to provenance metadata for diagnostics.

    Returns
    -------
    GraphValidationResult
        Validation results with deterministic ordering.
    """
    provider = schema_provider
    if validate_schema and provider is None:
        provider = _SCHEMA_PROVIDER_FACTORY()

    inputs = _collect_validation_inputs(nodes)

    errors: list[GraphValidationIssue] = [
        *inputs.materialize_issues,
        *inputs.dataset_issues,
        *inputs.artifact_issues,
        *inputs.saver_issues,
    ]
    warnings: list[GraphValidationIssue] = []
    errors.extend(_tag_type_issues(nodes))
    errors.extend(_duplicate_node_name_issues(nodes))
    errors.extend(_cpg2_naming_issues(nodes))
    errors.extend(_cpg_aggregator_dependency_issues(nodes))
    anchor_errors, anchor_warnings = _target_anchor_tag_issues(nodes)
    errors.extend(anchor_errors)
    errors.extend(_semantic_tag_issues(nodes))
    errors.extend(
        _orphan_saver_issues(
            nodes=nodes,
            materialize_nodes_by_target=inputs.materialize_nodes_by_target,
        )
    )
    errors.extend(_async_node_issues(nodes))
    errors.extend(_duplicate_materialize_issues(inputs.materialize_nodes_by_target))
    if validate_schema and provider is not None:
        schema_issues = _unknown_schema_issues(
            provider=provider,
            produced_table_to_target=inputs.saver_table_to_target,
        )
        errors.extend(issue for issue in schema_issues if issue.severity == "error")
        warnings.extend(issue for issue in schema_issues if issue.severity == "warning")

    warnings.extend(anchor_warnings)

    derived_deps: dict[str, tuple[str, ...]] | None = None
    if not any(issue.code == "duplicate_materialize" for issue in errors):
        derived_deps = _derive_target_dependencies(nodes, node_to_target=inputs.node_to_target)
        cycles = _find_cycles(derived_deps)
        errors.extend(
            GraphValidationIssue(
                severity="error",
                code="cycle_detected",
                message="Target dependency cycle detected: " + " -> ".join(cycle),
            )
            for cycle in cycles
        )

    if enforce_compute_io_purity:
        io_issues = _compute_io_purity_issues(nodes)
        errors.extend(i for i in io_issues if i.severity == "error")
        warnings.extend(i for i in io_issues if i.severity == "warning")

    if module_provenance:
        errors = _attach_issue_provenance(
            errors,
            nodes=nodes,
            module_provenance=module_provenance,
        )
        warnings = _attach_issue_provenance(
            warnings,
            nodes=nodes,
            module_provenance=module_provenance,
        )

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
    "validate_nodes",
    "validation_result_to_json",
]
