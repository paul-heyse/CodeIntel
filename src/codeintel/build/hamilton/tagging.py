"""Canonical Hamilton tagging helpers for build nodes."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import ParamSpec, Protocol, TypeVar, cast

from hamilton.function_modifiers import schema as h_schema
from hamilton.function_modifiers import tag as h_tag

from codeintel.build.hamilton.tag_spec import (
    NodeType,
    TagKey,
    TagSpec,
    TagValue,
    tag_spec_from_tags,
    validate_tag_spec,
)
from codeintel.build.schemas import get_schema_provider
from codeintel.core.hamilton import tags as ht
from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS
from codeintel.core.schemas.primitives import TableSchema

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]


class _HamiltonTagKwargs(Protocol):
    domain: TagValue
    target: TagValue
    table_key: TagValue
    artifact: TagValue
    artifact_path_template: TagValue
    node_type: TagValue
    output_kind: TagValue
    semantic_id: TagValue
    entity: TagValue
    grain: TagValue
    mcp_visible: TagValue
    tools: TagValue
    target_resources: TagValue
    target_execution: TagValue
    target_parameters: TagValue
    target_estimated_duration_ms: TagValue
    target_spec_version: TagValue


class _TagDecoratorFactory(Protocol):
    def __call__(
        self,
        *,
        target_: str | Collection[str] | None = None,
        **tags: TagValue,
    ) -> object: ...


_TAG_DECORATOR_FACTORY = cast("_TagDecoratorFactory", h_tag)


def apply_tags[TCallable](
    fn: TCallable,
    *,
    tags: Mapping[TagKey, TagValue],
) -> TCallable:
    """Apply canonical tag metadata to a callable.

    Returns
    -------
    TCallable
        Callable wrapped with tag metadata.
    """
    if not tags:
        return fn
    tags_kwargs: dict[TagKey, TagValue] = dict(tags)
    decorator = _tag(cast("_HamiltonTagKwargs", tags_kwargs), target_=None)
    return cast("TCallable", decorator(cast("Callable[..., object]", fn)))


def tag_from_spec(
    spec: TagSpec,
    *,
    target_: str | Collection[str] | None = None,
) -> Decorator[P, R]:
    """Create a Hamilton tag decorator from a TagSpec.

    Returns
    -------
    Decorator[P, R]
        Decorator applying TagSpec metadata.
    """
    validate_tag_spec(spec)
    tags = spec.to_tags()
    return _tag(cast("_HamiltonTagKwargs", tags), target_=target_)


def _tag(
    tags: _HamiltonTagKwargs,
    *,
    target_: str | Collection[str] | None,
) -> Decorator[P, R]:
    tags_kwargs = cast("dict[TagKey, TagValue]", tags)
    if target_ is None:
        return cast("Decorator[P, R]", _TAG_DECORATOR_FACTORY(**tags_kwargs))
    return cast("Decorator[P, R]", _TAG_DECORATOR_FACTORY(**tags_kwargs, target_=target_))


def tag_compute(
    *,
    domain: str,
    target: str | None = None,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a compute node with canonical build tags.

    Returns
    -------
    Decorator[P, R]
        Decorator applying compute node tags.
    """
    spec = TagSpec.for_compute(domain=domain, target=target, extra_tags=extra_tags)
    return tag_from_spec(spec, target_=target_)


def tag_materialize(
    *,
    domain: str,
    target: str | None = None,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a materialize node with canonical build tags.

    Returns
    -------
    Decorator[P, R]
        Decorator applying materialize node tags.
    """
    spec = TagSpec.for_materialize(domain=domain, target=target, extra_tags=extra_tags)
    return tag_from_spec(spec, target_=target_)


def tag_dataset(
    *,
    domain: str,
    target: str | None = None,
    table_key: str,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a dataset-producing node with canonical build tags.

    Returns
    -------
    Decorator[P, R]
        Decorator applying dataset node tags.
    """
    merged_tags: dict[TagKey, TagValue] = {cast("TagKey", ht.TAG_OUTPUT_KIND): ht.OUTPUT_KIND_TABLE}
    if extra_tags:
        merged_tags = {**merged_tags, **extra_tags}
    spec = TagSpec.for_dataset(
        domain=domain, target=target, table_key=table_key, extra_tags=merged_tags
    )
    tag_decorator = tag_from_spec(spec, target_=target_)
    schema_decorator = _schema_output_decorator(table_key)
    if schema_decorator is None:
        return tag_decorator
    schema_tag = cast("Decorator[P, R]", schema_decorator)

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = schema_tag(fn)
        return tag_decorator(tagged)

    return apply


def tag_artifact(
    *,
    domain: str,
    target: str | None = None,
    artifact: str,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag an artifact-producing node with canonical build tags.

    Returns
    -------
    Decorator[P, R]
        Decorator applying artifact node tags.
    """
    spec = TagSpec.for_artifact(
        domain=domain,
        target=target,
        artifact_name=artifact,
        extra_tags=extra_tags,
    )
    return tag_from_spec(spec, target_=target_)


def tag_tool(
    *,
    domain: str,
    target: str | None = None,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a tool/external-boundary node with canonical build tags.

    Returns
    -------
    Decorator[P, R]
        Decorator applying tool node tags.
    """
    spec = TagSpec.for_tool(domain=domain, target=target, extra_tags=extra_tags)
    return tag_from_spec(spec, target_=target_)


def tag_loader_query(
    *,
    domain: str,
    target: str | None = None,
    table_key: str,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a loader/query node with canonical build tags.

    Returns
    -------
    Decorator[P, R]
        Decorator applying loader query tags.
    """
    spec = TagSpec.for_loader_query(
        domain=domain,
        target=target,
        table_key=table_key,
        extra_tags=extra_tags,
    )
    return tag_from_spec(spec, target_=target_)


def _schema_output_decorator(table_key: str) -> Decorator[P, R] | None:
    schema = _resolve_table_schema(table_key)
    if schema is None:
        return None
    fields = tuple((column.name, column.type) for column in schema.columns)
    if not fields:
        return None
    tag_value = ",".join(f"{name}={dtype}" for name, dtype in fields)
    decorator = h_tag(
        bypass_reserved_namespaces_=True,
        **{h_schema.INTERNAL_SCHEMA_OUTPUT_KEY: tag_value},
    )
    return cast("Decorator[P, R]", decorator)


def _resolve_table_schema(table_key: str) -> TableSchema | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        return OUTPUT_TABLE_SCHEMAS.get(table_key)
    schema = provider.get_table_schema(table_key)
    return schema or OUTPUT_TABLE_SCHEMAS.get(table_key)


def tag_helper(
    *,
    domain: str,
    target: str | None = None,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a helper node with canonical build tags.

    Returns
    -------
    Decorator[P, R]
        Decorator applying helper node tags.
    """
    spec = TagSpec.for_helper(domain=domain, target=target, extra_tags=extra_tags)
    return tag_from_spec(spec, target_=target_)


__all__ = [
    "Decorator",
    "NodeType",
    "TagKey",
    "TagSpec",
    "TagValue",
    "apply_tags",
    "tag_artifact",
    "tag_compute",
    "tag_dataset",
    "tag_from_spec",
    "tag_helper",
    "tag_loader_query",
    "tag_materialize",
    "tag_spec_from_tags",
    "tag_tool",
    "validate_tag_spec",
]
