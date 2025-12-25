"""Internal module attachment helpers for dynamic node generation."""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Protocol, cast

from hamilton.function_modifiers import tag as h_tag

from codeintel.build.hamilton.tagging import TagAttachContext, TagKey, TagValue, build_tag_kwargs


class _NamedCallable(Protocol):
    """Protocol for callables with mutable naming metadata."""

    __name__: str
    __module__: str
    __dict__: dict[str, object]


def attach_node(module: ModuleType, *, node_name: str, fn: object) -> None:
    """Attach a callable to a module under a stable node name."""
    named = cast("_NamedCallable", fn)
    named.__name__ = node_name
    named.__module__ = module.__name__
    setattr(module, node_name, fn)


def _extract_tag_metadata(fn: object) -> dict[str, TagValue]:
    tags: dict[str, TagValue] = {}
    meta = getattr(fn, "__dict__", None)
    if not isinstance(meta, dict):
        return tags
    decorators = meta.get("decorate_nodes")
    if not isinstance(decorators, list):
        return tags
    for decorator in decorators:
        decorated_tags = getattr(decorator, "tags", None)
        if not isinstance(decorated_tags, dict):
            continue
        for key, value in decorated_tags.items():
            if not isinstance(key, str) or key in tags:
                continue
            if isinstance(value, str) or (isinstance(value, list) and value and all(isinstance(item, str) for item in value)):
                tags[key] = value
    return tags


def _merge_extra_tags(
    context: TagAttachContext | None,
    extra_tags: Mapping[TagKey, TagValue] | None,
) -> dict[TagKey, TagValue]:
    merged: dict[TagKey, TagValue] = {}
    if context is not None and context.extra_tags is not None:
        merged.update(context.extra_tags)
    if extra_tags is not None:
        merged.update(extra_tags)
    return merged


def tagged_attach_node(
    module: ModuleType,
    *,
    node_name: str,
    fn: object,
    context: TagAttachContext | None = None,
    node_type: str | None = None,
    domain: str | None = None,
    target: str | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> None:
    """Attach a callable to a module while enforcing canonical tag metadata."""
    if not callable(fn):
        msg = f"tagged_attach_node requires a callable, got {type(fn)}"
        raise TypeError(msg)

    existing_tags = _extract_tag_metadata(fn)
    merged_extra_tags = _merge_extra_tags(context, extra_tags)

    node_type_from_tags = existing_tags.get("node_type")
    node_type_from_extra = merged_extra_tags.get("node_type")
    resolved_node_type = node_type or (context.node_type if context is not None else None)
    if resolved_node_type is None and isinstance(node_type_from_extra, str):
        resolved_node_type = node_type_from_extra
        merged_extra_tags.pop(cast("TagKey", "node_type"), None)
    if resolved_node_type is None and isinstance(node_type_from_tags, str):
        resolved_node_type = node_type_from_tags
    if resolved_node_type is None:
        msg = f"tagged_attach_node requires node_type for {node_name}"
        raise ValueError(msg)

    resolved_domain = domain if domain is not None else (context.domain if context else None)
    resolved_target = target if target is not None else (context.target if context else None)
    merged_extra_tags = merged_extra_tags or None
    desired_tags = build_tag_kwargs(
        node_type=resolved_node_type,
        domain=resolved_domain,
        target=resolved_target,
        extra_tags=merged_extra_tags,
    )
    tags_to_add = {
        key: value for key, value in desired_tags.items() if key not in existing_tags
    }
    tagged_fn = fn
    if tags_to_add:
        tagged_fn = h_tag(**tags_to_add)(fn)
    attach_node(module, node_name=node_name, fn=tagged_fn)


__all__ = ["attach_node", "tagged_attach_node"]
