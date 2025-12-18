"""Canonical Hamilton tagging helpers for build nodes.

This module provides small wrappers around Hamilton's ``@tag`` decorator using the canonical tag
keys and node type values defined in ``codeintel.hamilton.tags``.

The intent is to prevent "tag drift" by making common tagging patterns easy and consistent while
still allowing controlled extension via ``extra_tags``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from types import EllipsisType
from typing import ParamSpec, TypeVar, cast

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

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]


def _merge_extra_tags(
    base: dict[str, object],
    extra_tags: Mapping[str, object] | None,
) -> dict[str, object]:
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
    extra_tags: Mapping[str, object] | None,
) -> dict[str, object]:
    base: dict[str, object] = {TAG_NODE_TYPE: node_type}
    if domain is not None:
        base[TAG_DOMAIN] = domain
    if target is not None:
        base[TAG_TARGET] = target
    return _merge_extra_tags(base, extra_tags)


def tag_node(
    *,
    node_type: str,
    domain: str | None = None,
    target: str | None = None,
    table_key: str | None = None,
    artifact: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    """Return a Hamilton ``@tag`` decorator with canonical keys.

    Parameters
    ----------
    node_type
        Canonical node type value (e.g. ``NODE_TYPE_COMPUTE``).
    domain
        Optional domain identifier for the node (e.g. ``"analytics"``).
    target
        Optional target name for the node.
    table_key
        Optional dataset table key (required for dataset nodes).
    artifact
        Optional artifact name (required for artifact nodes).
    extra_tags
        Additional tags to merge (later wins on key conflict).

    Returns
    -------
    Callable
        Decorator that tags the underlying Hamilton node.
    """
    tags = _build_common_tags(node_type=node_type, domain=domain, target=target, extra_tags=extra_tags)
    if table_key is not None:
        tags[TAG_TABLE_KEY] = table_key
    if artifact is not None:
        tags[TAG_ARTIFACT] = artifact
    if target_ is None:
        return cast("Decorator[P, R]", h_tag(**tags))
    return cast("Decorator[P, R]", h_tag(**tags, target_=target_))


def tag_compute(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_COMPUTE,
        domain=domain,
        target=target,
        target_=target_,
        extra_tags=extra_tags,
    )


def tag_materialize(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_MATERIALIZE,
        domain=domain,
        target=target,
        target_=target_,
        extra_tags=extra_tags,
    )


def tag_dataset(
    *,
    domain: str | None = None,
    target: str | None = None,
    table_key: str,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_DATASET,
        domain=domain,
        target=target,
        table_key=table_key,
        target_=target_,
        extra_tags=extra_tags,
    )


def tag_artifact(
    *,
    domain: str | None = None,
    target: str | None = None,
    artifact: str,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_ARTIFACT,
        domain=domain,
        target=target,
        artifact=artifact,
        target_=target_,
        extra_tags=extra_tags,
    )


def tag_tool(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_TOOL, domain=domain, target=target, target_=target_, extra_tags=extra_tags
    )


def tag_loader_query(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_LOADER_QUERY,
        domain=domain,
        target=target,
        target_=target_,
        extra_tags=extra_tags,
    )


def tag_loader_dataframe(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_LOADER_DATAFRAME,
        domain=domain,
        target=target,
        target_=target_,
        extra_tags=extra_tags,
    )


def tag_helper(
    *,
    domain: str | None = None,
    target: str | None = None,
    target_: str | Collection[str] | EllipsisType | None = None,
    extra_tags: Mapping[str, object] | None = None,
) -> Decorator[P, R]:
    return tag_node(
        node_type=NODE_TYPE_HELPER,
        domain=domain,
        target=target,
        target_=target_,
        extra_tags=extra_tags,
    )


__all__ = [
    "Decorator",
    "tag_artifact",
    "tag_compute",
    "tag_dataset",
    "tag_helper",
    "tag_loader_dataframe",
    "tag_loader_query",
    "tag_materialize",
    "tag_node",
    "tag_tool",
]
