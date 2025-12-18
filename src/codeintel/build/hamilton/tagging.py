"""Canonical Hamilton tagging helpers for build nodes.

This module provides small wrappers around Hamilton's ``@tag`` decorator using the canonical tag
keys and node type values defined in ``codeintel.hamilton.tags``.

The intent is to prevent "tag drift" by making common tagging patterns easy and consistent while
still allowing controlled extension via ``extra_tags``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from hamilton.function_modifiers import tag as h_tag

from codeintel.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_COMPUTE,
    NODE_TYPE_DATASET,
    NODE_TYPE_HELPER,
    NODE_TYPE_LOADER_DATAFRAME,
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_MATERIALIZE,
    NODE_TYPE_TOOL,
    TAG_ARTIFACT,
    TAG_DOMAIN,
    TAG_NODE_TYPE,
    TAG_TABLE_KEY,
    TAG_TARGET,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from types import EllipsisType

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]
TagValue = str | list[str]


def _merge_extra_tags(
    base: dict[str, TagValue],
    extra_tags: Mapping[str, TagValue] | None,
) -> dict[str, TagValue]:
    if extra_tags is None:
        return base
    merged = dict(base)
    merged.update(extra_tags)
    return merged


def _build_common_tags(
    *,
    node_type: str,
    domain: str | None,
    target: str | None,
    extra_tags: Mapping[str, TagValue] | None,
) -> dict[str, TagValue]:
    base: dict[str, TagValue] = {TAG_NODE_TYPE: node_type}
    if domain is not None:
        base[TAG_DOMAIN] = domain
    if target is not None:
        base[TAG_TARGET] = target
    return _merge_extra_tags(base, extra_tags)


def _tag(
    tags: dict[str, TagValue],
    *,
    target_: str | Collection[str] | EllipsisType | None,
) -> Decorator[P, R]:
    if target_ is None:
        return cast("Decorator[P, R]", h_tag(**tags))
    return cast("Decorator[P, R]", h_tag(**tags, target_=target_))


def tag_compute(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a compute node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical compute tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_COMPUTE,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    return _tag(tags, target_=target_)


def tag_materialize(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a materialize node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical materialize tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_MATERIALIZE,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    return _tag(tags, target_=target_)


def tag_dataset(
    *,
    domain: str | None = None,
    target: str | None = None,
    table_key: str,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a dataset-producing node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical dataset tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_DATASET,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    tags[TAG_TABLE_KEY] = table_key
    return _tag(tags, target_=target_)


def tag_artifact(
    *,
    domain: str | None = None,
    target: str | None = None,
    artifact: str,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag an artifact-producing node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical artifact tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_ARTIFACT,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    tags[TAG_ARTIFACT] = artifact
    return _tag(tags, target_=target_)


def tag_tool(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a tool/external-boundary node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical tool tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_TOOL,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    return _tag(tags, target_=target_)


def tag_loader_query(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a loader/query node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical loader-query tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_LOADER_QUERY,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    return _tag(tags, target_=target_)


def tag_loader_dataframe(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a loader/dataframe node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical loader-dataframe tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_LOADER_DATAFRAME,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    return _tag(tags, target_=target_)


def tag_helper(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, TagValue] | None = None,
) -> Decorator[P, R]:
    """Tag a helper node with canonical build tags.

    Returns
    -------
    Decorator
        Hamilton decorator that applies the canonical helper tags.
    """
    tags = _build_common_tags(
        node_type=NODE_TYPE_HELPER,
        domain=domain,
        target=target,
        extra_tags=extra_tags,
    )
    return _tag(tags, target_=target_)


__all__ = [
    "Decorator",
    "tag_artifact",
    "tag_compute",
    "tag_dataset",
    "tag_helper",
    "tag_loader_dataframe",
    "tag_loader_query",
    "tag_materialize",
    "tag_tool",
]
