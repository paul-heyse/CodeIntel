"""Hamilton tag decorator helpers for non-build layers."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Protocol, cast

from hamilton.function_modifiers import tag as h_tag

TagValue = str | list[str]
TagInputValue = TagValue | bool


class _TagDecoratorFactory(Protocol):
    def __call__(
        self,
        *,
        target_: str | Collection[str] | None = None,
        **tags: TagValue,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...


_TAG_DECORATOR_FACTORY = cast("_TagDecoratorFactory", h_tag)


def apply_raw_tags[TFunc: Callable[..., object]](
    fn: TFunc,
    *,
    tags: Mapping[str, TagInputValue],
    target_: str | Collection[str] | None = None,
) -> TFunc:
    """Apply Hamilton tag metadata without key validation.

    Parameters
    ----------
    fn
        Callable to decorate with tags.
    tags
        Tag mapping applied to the callable.
    target_
        Optional target filter for Hamilton tags.

    Returns
    -------
    TFunc
        Tagged callable.
    """
    tag_kwargs = {key: _normalize_tag_value(value) for key, value in tags.items()}
    decorator = (
        _TAG_DECORATOR_FACTORY(**tag_kwargs)
        if target_ is None
        else _TAG_DECORATOR_FACTORY(**tag_kwargs, target_=target_)
    )
    return cast("TFunc", decorator(cast("Callable[..., object]", fn)))


def _normalize_tag_value(value: TagInputValue) -> TagValue:
    if isinstance(value, bool):
        return "1" if value else "0"
    return value


def extract_tag_decorator_tags(decorator: object) -> dict[str, object] | None:
    """Extract tags from a Hamilton tag decorator instance.

    Parameters
    ----------
    decorator
        Potential Hamilton tag decorator instance.

    Returns
    -------
    dict[str, object] | None
        Tag mapping when the decorator is a Hamilton tag.
    """
    if not isinstance(decorator, h_tag):
        return None
    tags = getattr(decorator, "tags", None)
    return dict(tags) if isinstance(tags, dict) else None


__all__ = [
    "TagInputValue",
    "TagValue",
    "apply_raw_tags",
    "extract_tag_decorator_tags",
]
