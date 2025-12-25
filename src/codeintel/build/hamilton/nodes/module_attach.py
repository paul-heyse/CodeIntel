"""Internal module attachment helpers for dynamic node generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Protocol, cast

from codeintel.build.hamilton.tag_spec import TagKey, TagSpec, TagValue, tag_spec_from_tags
from codeintel.build.hamilton.tagging import apply_tags, validate_tag_spec


class _NamedCallable(Protocol):
    """Protocol for callables with mutable naming metadata."""

    __name__: str
    __module__: str
    __dict__: dict[str, object]


def attach_node(module: ModuleType, *, node_name: str, fn: Callable[..., object]) -> None:
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
            if isinstance(value, str) or (
                isinstance(value, list) and value and all(isinstance(item, str) for item in value)
            ):
                tags[key] = value
    return tags


def tagged_attach_node(
    module: ModuleType,
    *,
    node_name: str,
    fn: Callable[..., object],
    tag_spec: TagSpec,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> None:
    """Attach a callable to a module while enforcing canonical tag metadata.

    Raises
    ------
    TypeError
        If the provided object is not callable.
    ValueError
        If required tags are missing or existing tags conflict with TagSpec.
    """
    if not callable(fn):
        msg = f"tagged_attach_node expects a callable for {node_name}"
        raise TypeError(msg)

    resolved_spec = tag_spec.with_extra_tags(extra_tags)
    validate_tag_spec(resolved_spec)
    desired_tags = resolved_spec.to_tags()

    existing_tags = _extract_tag_metadata(fn)
    for key, value in desired_tags.items():
        if key in existing_tags and existing_tags[key] != value:
            msg = f"Conflicting tag {key} for {node_name}"
            raise ValueError(msg)

    tags_to_add = {key: value for key, value in desired_tags.items() if key not in existing_tags}
    tagged_fn = fn
    if tags_to_add:
        tagged_fn = apply_tags(fn, tags=tags_to_add)

    if existing_tags:
        existing_spec = tag_spec_from_tags(existing_tags)
        if existing_spec is not None and existing_spec.node_type != resolved_spec.node_type:
            msg = f"Existing node_type tag mismatch for {node_name}"
            raise ValueError(msg)

    attach_node(module, node_name=node_name, fn=tagged_fn)


__all__ = ["attach_node", "tagged_attach_node"]
