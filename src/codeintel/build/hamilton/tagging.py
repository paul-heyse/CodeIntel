"""Canonical Hamilton tagging helpers for build nodes."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import ParamSpec, Protocol, TypeVar, cast

from hamilton.function_modifiers import tag as h_tag

from codeintel.build.hamilton.tag_spec import (
    NodeType,
    TagKey,
    TagSpec,
    TagValue,
    tag_spec_from_tags,
    validate_tag_spec,
)

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
    """Apply canonical tag metadata to a callable."""
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
    """Create a Hamilton tag decorator from a TagSpec."""
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
    """Tag a compute node with canonical build tags."""
    spec = TagSpec.for_compute(domain=domain, target=target, extra_tags=extra_tags)
    return tag_from_spec(spec, target_=target_)


def tag_materialize(
    *,
    domain: str,
    target: str | None = None,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a materialize node with canonical build tags."""
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
    """Tag a dataset-producing node with canonical build tags."""
    spec = TagSpec.for_dataset(domain=domain, target=target, table_key=table_key, extra_tags=extra_tags)
    return tag_from_spec(spec, target_=target_)


def tag_artifact(
    *,
    domain: str,
    target: str | None = None,
    artifact: str,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag an artifact-producing node with canonical build tags."""
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
    """Tag a tool/external-boundary node with canonical build tags."""
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
    """Tag a loader/query node with canonical build tags."""
    spec = TagSpec.for_loader_query(
        domain=domain,
        target=target,
        table_key=table_key,
        extra_tags=extra_tags,
    )
    return tag_from_spec(spec, target_=target_)


def tag_loader_dataframe(
    *,
    domain: str,
    target: str | None = None,
    table_key: str,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a loader/dataframe node with canonical build tags."""
    spec = TagSpec.for_loader_dataframe(
        domain=domain,
        target=target,
        table_key=table_key,
        extra_tags=extra_tags,
    )
    return tag_from_spec(spec, target_=target_)


def tag_helper(
    *,
    domain: str,
    target: str | None = None,
    target_: str | Collection[str] | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a helper node with canonical build tags."""
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
    "tag_loader_dataframe",
    "tag_loader_query",
    "tag_materialize",
    "tag_spec_from_tags",
    "tag_tool",
    "validate_tag_spec",
]
