"""Compile OutputTarget specs from Hamilton DAG tags."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from codeintel.build.contracts import ArtifactSpec, OutputContract
from codeintel.build.hamilton.introspect import derive_target_outputs_from_savers
from codeintel.build.hamilton.runtime import HamiltonRuntime
from codeintel.build.hamilton.validate import validate_nodes
from codeintel.build.parameters import EMPTY_PARAMETERS, TargetParameters
from codeintel.build.resources import (
    DEFAULT_EXECUTION,
    DEFAULT_RESOURCES,
    IsolationKind,
    TargetExecution,
    TargetResources,
)
from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule
from codeintel.core.hamilton import tags as ht
from codeintel.core.schemas.table_registry import get_table_schema
from codeintel.storage.helpers.table_key import validate_table_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hamilton.driver import Driver
    from hamilton.node import Node

    from codeintel.build.hamilton.introspect import DerivedTargetOutputs
    from codeintel.build.hamilton.validate import GraphValidationIssue
    from codeintel.core.schemas.primitives import TableSchema


@dataclass(frozen=True, slots=True)
class TargetSpecOverride:
    """Small override layer for truly non-derivable settings."""

    resources: TargetResources | None = None
    execution: TargetExecution | None = None
    parameters: TargetParameters | None = None
    description: str | None = None


class TargetSpecError(RuntimeError):
    """Error raised when target spec compilation fails."""

    @classmethod
    def graph_validation(cls, errors: Sequence[GraphValidationIssue]) -> TargetSpecError:
        lines = ["Graph validation errors:"]
        lines.extend(f"- {issue.message}" for issue in errors)
        return cls("\n".join(lines))

    @classmethod
    def duplicate_anchor(cls, target_name: str) -> TargetSpecError:
        return cls(f"Duplicate t__ materialize node for target {target_name}")

    @classmethod
    def no_targets(cls) -> TargetSpecError:
        return cls("No build targets discovered in Hamilton graph (missing t__ nodes?)")

    @classmethod
    def missing_domain(cls, target_name: str) -> TargetSpecError:
        return cls(f"Target {target_name} missing domain tag")

    @classmethod
    def invalid_domain(cls, target_name: str, domain: str) -> TargetSpecError:
        return cls(f"Target {target_name} has invalid domain tag: {domain!r}")

    @classmethod
    def missing_description(cls, target_name: str) -> TargetSpecError:
        return cls(f"Target {target_name} must have a docstring summary (or override)")

    @classmethod
    def invalid_spec_version(cls, target_name: str, spec_version: object) -> TargetSpecError:
        return cls(f"Target {target_name} missing/invalid spec version tag: {spec_version!r}")

    @classmethod
    def missing_tag(cls, target_name: str, key: str) -> TargetSpecError:
        return cls(f"Target {target_name} missing {key} tag")

    @classmethod
    def invalid_tag_json(cls, target_name: str, key: str) -> TargetSpecError:
        return cls(f"Target {target_name} has invalid {key} tag JSON")

    @classmethod
    def invalid_tag_value(cls, target_name: str, key: str) -> TargetSpecError:
        return cls(f"Target {target_name} has invalid {key} tag")

    @classmethod
    def missing_table_schema(cls, target_name: str, table_key: str) -> TargetSpecError:
        return cls(f"Missing TableSchema for {target_name} output: {table_key}")


def _node_docstring(node: Node) -> str:
    for attr in ("documentation", "doc_string", "doc", "description"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    fn = getattr(node, "callable", None) or getattr(node, "func", None)
    if callable(fn):
        doc = fn.__doc__ or ""
        if doc.strip():
            return doc.strip()

    return ""


def _summary(doc: str) -> str:
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _parse_json_tag(
    *,
    tags: Mapping[str, object],
    key: str,
    target_name: str,
    strict: bool,
) -> dict[str, object] | None:
    raw = tags.get(key)
    if raw is None:
        if strict:
            raise TargetSpecError.missing_tag(target_name, key)
        return None
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            if strict:
                raise TargetSpecError.invalid_tag_json(target_name, key) from exc
            return None
        if isinstance(parsed, dict):
            return dict(parsed)
    if strict:
        raise TargetSpecError.invalid_tag_value(target_name, key)
    return None


_VALID_DOMAINS: frozenset[str] = frozenset({"analytics", "export", "graphs", "ingestion"})
_VALID_ISOLATIONS: frozenset[str] = frozenset({"none", "process", "thread"})


def _parse_bool(
    value: object,
    *,
    default: bool,
    strict: bool,
    target_name: str,
    tag_key: str,
) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if strict:
        raise TargetSpecError.invalid_tag_value(target_name, tag_key)
    return default


def _parse_int(
    value: object,
    *,
    default: int,
    strict: bool,
    target_name: str,
    tag_key: str,
) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if value is None:
        return default
    if strict:
        raise TargetSpecError.invalid_tag_value(target_name, tag_key)
    return default


def _parse_optional_int(
    value: object,
    *,
    default: int | None,
    strict: bool,
    target_name: str,
    tag_key: str,
) -> int | None:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if strict:
        raise TargetSpecError.invalid_tag_value(target_name, tag_key)
    return default


def _parse_isolation(
    value: object,
    *,
    default: IsolationKind,
    strict: bool,
    target_name: str,
) -> IsolationKind:
    if isinstance(value, str) and value in _VALID_ISOLATIONS:
        return cast("IsolationKind", value)
    if value is None:
        return default
    if strict:
        raise TargetSpecError.invalid_tag_value(target_name, ht.TAG_TARGET_EXECUTION)
    return default


def _parse_tools(
    value: object,
    *,
    default: tuple[str, ...],
    strict: bool,
    target_name: str,
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    if strict:
        raise TargetSpecError.invalid_tag_value(target_name, ht.TAG_TARGET_RESOURCES)
    return default


def _derive_tools_from_nodes(
    nodes: Mapping[str, Node],
    *,
    target_name: str,
) -> tuple[str, ...]:
    tools: set[str] = set()
    for node in nodes.values():
        tags = getattr(node, "tags", None)
        if not isinstance(tags, dict):
            continue
        if tags.get(ht.TAG_NODE_TYPE) != ht.NODE_TYPE_TOOL:
            continue
        if tags.get(ht.TAG_TARGET) != target_name:
            continue
        raw = tags.get(ht.TAG_TOOLS)
        if isinstance(raw, str) and raw:
            tools.add(raw)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str) and item:
                    tools.add(item)
    return tuple(sorted(tools))


def _resources_from_tags(
    *,
    tags: Mapping[str, object],
    target_name: str,
    strict: bool,
) -> TargetResources:
    obj = _parse_json_tag(
        tags=tags,
        key=ht.TAG_TARGET_RESOURCES,
        target_name=target_name,
        strict=strict,
    )
    if obj is None:
        return DEFAULT_RESOURCES

    tracker = _parse_bool(
        obj.get("tracker"),
        default=DEFAULT_RESOURCES.tracker,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_RESOURCES,
    )
    modules = _parse_bool(
        obj.get("modules"),
        default=DEFAULT_RESOURCES.modules,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_RESOURCES,
    )
    gateway = _parse_bool(
        obj.get("gateway"),
        default=DEFAULT_RESOURCES.gateway,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_RESOURCES,
    )
    tools = _parse_tools(
        obj.get("tools"),
        default=DEFAULT_RESOURCES.tools,
        strict=strict,
        target_name=target_name,
    )
    return TargetResources(
        tracker=tracker,
        modules=modules,
        gateway=gateway,
        tools=tools,
    )


def _execution_from_tags(
    *,
    tags: Mapping[str, object],
    target_name: str,
    strict: bool,
) -> TargetExecution:
    obj = _parse_json_tag(
        tags=tags,
        key=ht.TAG_TARGET_EXECUTION,
        target_name=target_name,
        strict=strict,
    )
    if obj is None:
        return DEFAULT_EXECUTION

    cpu_intensive = _parse_bool(
        obj.get("cpu_intensive"),
        default=DEFAULT_EXECUTION.cpu_intensive,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_EXECUTION,
    )
    io_intensive = _parse_bool(
        obj.get("io_intensive"),
        default=DEFAULT_EXECUTION.io_intensive,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_EXECUTION,
    )
    memory_intensive = _parse_bool(
        obj.get("memory_intensive"),
        default=DEFAULT_EXECUTION.memory_intensive,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_EXECUTION,
    )
    max_runtime_ms = _parse_int(
        obj.get("max_runtime_ms"),
        default=DEFAULT_EXECUTION.max_runtime_ms,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_EXECUTION,
    )
    isolation = _parse_isolation(
        obj.get("isolation"),
        default=DEFAULT_EXECUTION.isolation,
        strict=strict,
        target_name=target_name,
    )
    supports_incremental = _parse_bool(
        obj.get("supports_incremental"),
        default=DEFAULT_EXECUTION.supports_incremental,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_EXECUTION,
    )
    max_parallelism = _parse_optional_int(
        obj.get("max_parallelism"),
        default=DEFAULT_EXECUTION.max_parallelism,
        strict=strict,
        target_name=target_name,
        tag_key=ht.TAG_TARGET_EXECUTION,
    )
    return TargetExecution(
        cpu_intensive=cpu_intensive,
        io_intensive=io_intensive,
        memory_intensive=memory_intensive,
        max_runtime_ms=max_runtime_ms,
        isolation=isolation,
        supports_incremental=supports_incremental,
        max_parallelism=max_parallelism,
    )


def _parameters_from_tags(
    *,
    tags: Mapping[str, object],
    target_name: str,
    strict: bool,
) -> TargetParameters:
    obj = _parse_json_tag(
        tags=tags,
        key=ht.TAG_TARGET_PARAMETERS,
        target_name=target_name,
        strict=strict,
    )
    if obj is None:
        return EMPTY_PARAMETERS
    params = {str(key): value for key, value in obj.items()}
    return TargetParameters(params)


def _is_target_anchor(node: Node) -> bool:
    tags = getattr(node, "tags", None)
    if not isinstance(tags, dict):
        return False
    return (
        tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_MATERIALIZE
        and isinstance(tags.get(ht.TAG_TARGET), str)
        and isinstance(tags.get(ht.TAG_DOMAIN), str)
    )


def _resolve_domain(
    *,
    tags: Mapping[str, object],
    target_name: str,
    strict: bool,
) -> TargetModule:
    domain = tags.get(ht.TAG_DOMAIN)
    if not isinstance(domain, str) or not domain:
        if strict:
            raise TargetSpecError.missing_domain(target_name)
        return cast("TargetModule", "ingestion")
    if strict and domain not in _VALID_DOMAINS:
        raise TargetSpecError.invalid_domain(target_name, domain)
    return cast("TargetModule", domain)


def _resolve_description(
    node: Node,
    *,
    target_name: str,
    override: TargetSpecOverride | None,
    strict: bool,
) -> str:
    doc = _node_docstring(node)
    description = _summary(doc)
    if override and override.description:
        description = override.description
    if strict and not description:
        raise TargetSpecError.missing_description(target_name)
    return description


def _validate_spec_version(
    *,
    tags: Mapping[str, object],
    target_name: str,
    strict: bool,
) -> None:
    spec_version = tags.get(ht.TAG_TARGET_SPEC_VERSION)
    if strict and spec_version != "1":
        raise TargetSpecError.invalid_spec_version(target_name, spec_version)


def _collect_target_anchors(
    nodes: Mapping[str, Node],
    *,
    strict: bool,
) -> dict[str, Node]:
    anchors: dict[str, Node] = {}
    for node in nodes.values():
        if not _is_target_anchor(node):
            continue
        tags = node.tags
        target_name = tags.get(ht.TAG_TARGET)
        if not isinstance(target_name, str):
            continue
        if target_name in anchors:
            raise TargetSpecError.duplicate_anchor(target_name)
        anchors[target_name] = node

    if strict and not anchors:
        raise TargetSpecError.no_targets()
    return anchors


def _resolve_table_schemas(
    *,
    target_name: str,
    table_keys: tuple[str, ...],
    strict: bool,
) -> tuple[TableSchema, ...]:
    tables: list[TableSchema] = []
    seen: set[str] = set()
    for key in table_keys:
        validate_table_key(key)
        if key in seen:
            msg = f"Duplicate table_key in target spec: {key}"
            raise ValueError(msg)
        seen.add(key)
        schema = get_table_schema(key)
        if schema is None:
            if strict:
                raise TargetSpecError.missing_table_schema(target_name, key)
            continue
        tables.append(schema)

    return tuple(tables)


def _artifact_specs(
    *,
    target_name: str,
    artifact_names: tuple[str, ...],
    templates_by_target: Mapping[str, Mapping[str, str]],
) -> tuple[ArtifactSpec, ...]:
    template_map = templates_by_target.get(target_name, {})
    specs: list[ArtifactSpec] = []
    for name in artifact_names:
        template = template_map.get(name)
        if not isinstance(template, str) or not template:
            msg = f"Missing artifact path template for {target_name}.{name}"
            raise RuntimeError(msg)
        specs.append(ArtifactSpec(name=name, path_template=template))
    return tuple(sorted(specs, key=lambda spec: spec.name))


def _build_output_target(
    *,
    target_name: str,
    node: Node,
    derived_outputs: DerivedTargetOutputs,
    derived_tools: Mapping[str, tuple[str, ...]],
    overrides_by_target: Mapping[str, TargetSpecOverride],
    strict: bool,
) -> OutputTarget:
    tags = cast("dict[str, object]", node.tags)
    override = overrides_by_target.get(target_name)

    domain = _resolve_domain(tags=tags, target_name=target_name, strict=strict)
    description = _resolve_description(
        node,
        target_name=target_name,
        override=override,
        strict=strict,
    )
    _validate_spec_version(tags=tags, target_name=target_name, strict=strict)

    resources = _resources_from_tags(tags=tags, target_name=target_name, strict=strict)
    execution = _execution_from_tags(tags=tags, target_name=target_name, strict=strict)
    parameters = _parameters_from_tags(tags=tags, target_name=target_name, strict=strict)

    if override and override.resources is not None:
        resources = override.resources
    elif not resources.tools:
        tools = derived_tools.get(target_name)
        if tools:
            resources = TargetResources(
                tracker=resources.tracker,
                modules=resources.modules,
                gateway=resources.gateway,
                tools=tools,
            )
    if override and override.execution is not None:
        execution = override.execution
    if override and override.parameters is not None:
        parameters = override.parameters

    table_keys = derived_outputs.datasets_by_target.get(target_name, ())
    tables = _resolve_table_schemas(
        target_name=target_name,
        table_keys=table_keys,
        strict=strict,
    )

    artifact_names = derived_outputs.artifacts_by_target.get(target_name, ())
    artifacts = _artifact_specs(
        target_name=target_name,
        artifact_names=artifact_names,
        templates_by_target=derived_outputs.artifact_templates_by_target,
    )

    contract = OutputContract(tables=tables, artifacts=artifacts)
    return OutputTarget(
        name=target_name,
        module=domain,
        contract=contract,
        dependencies=(),
        resources=resources,
        execution=execution,
        parameters=parameters,
        description=description,
    )


def compile_output_targets_from_driver(
    driver: Driver,
    *,
    overrides_by_target: Mapping[str, TargetSpecOverride] | None = None,
    strict: bool = True,
) -> tuple[OutputTarget, ...]:
    """Compile OutputTarget specs from a Hamilton Driver.

    Returns
    -------
    tuple[OutputTarget, ...]
        Sorted output targets derived from the driver graph.

    Raises
    ------
    TargetSpecError
        If validation fails or required target metadata is missing.
    """
    nodes = driver.graph.nodes
    overrides_by_target = overrides_by_target or MappingProxyType({})

    if strict:
        validation = validate_nodes(nodes, validate_schema=False)
        if validation.errors:
            lines = ["Graph validation errors:"]
            lines.extend(f"- {issue.message}" for issue in validation.errors)
            message = "\n".join(lines)
            raise TargetSpecError(message)

    runtime = HamiltonRuntime(dr=driver, graph=TargetGraph())
    derived_outputs = derive_target_outputs_from_savers(runtime)
    anchors = _collect_target_anchors(nodes, strict=strict)
    derived_tools = {
        target_name: _derive_tools_from_nodes(nodes, target_name=target_name)
        for target_name in anchors
    }

    results = [
        _build_output_target(
            target_name=target_name,
            node=anchors[target_name],
            derived_outputs=derived_outputs,
            derived_tools=derived_tools,
            overrides_by_target=overrides_by_target,
            strict=strict,
        )
        for target_name in sorted(anchors)
    ]
    return tuple(results)


__all__ = [
    "TargetSpecOverride",
    "compile_output_targets_from_driver",
]
