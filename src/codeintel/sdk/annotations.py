"""Stable wrappers for Hamilton annotation modifiers."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import ParamSpec, Protocol, TypeVar, cast

from hamilton.function_modifiers import schema as h_schema
from hamilton.function_modifiers import tag as h_tag

from codeintel.core.hamilton import semantic_tags as hst
from codeintel.core.hamilton import tags as ht

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]
TagValue = str | list[str]


class _TagDecoratorFactory(Protocol):
    def __call__(
        self,
        *,
        target_: str | Collection[str] | None = None,
        **tags: TagValue,
    ) -> object: ...


_ALLOWED_TAG_KEYS = frozenset(
    {
        ht.TAG_DOMAIN,
        ht.TAG_TARGET,
        ht.TAG_TABLE_KEY,
        ht.TAG_ARTIFACT,
        ht.TAG_ARTIFACT_PATH_TEMPLATE,
        ht.TAG_NODE_TYPE,
        ht.TAG_TARGET_RESOURCES,
        ht.TAG_TARGET_EXECUTION,
        ht.TAG_TARGET_PARAMETERS,
        ht.TAG_TARGET_ESTIMATED_DURATION_MS,
        ht.TAG_TARGET_SPEC_VERSION,
        ht.TAG_TOOLS,
        ht.TAG_OUTPUT_KIND,
        ht.TAG_SEMANTIC_ID,
        ht.TAG_ENTITY,
        ht.TAG_GRAIN,
        ht.TAG_MCP_VISIBLE,
        ht.TAG_LAYER,
        ht.TAG_KIND,
        ht.TAG_VERSION,
        ht.TAG_SCHEMA_REF,
        ht.TAG_ENTITY_KEYS,
        ht.TAG_JOIN_KEYS,
        ht.TAG_DTYPE,
        ht.TAG_STABILITY,
        ht.TAG_OWNER,
        ht.TAG_DESCRIPTION,
        ht.TAG_UNIT,
        ht.TAG_PII,
        ht.TAG_SOURCE_SYSTEM,
        ht.TAG_MATERIALIZATION,
        ht.TAG_MATERIALIZED_NAME,
        hst.TAG_DEFAULT_LIMIT,
        hst.TAG_DEFAULT_ORDER,
        hst.TAG_SEMANTIC_COLS,
        hst.TAG_SEMANTIC_DESC,
        hst.TAG_SEMANTIC_ENTITY,
        hst.TAG_SEMANTIC_GRAIN,
        hst.TAG_SEMANTIC_JOINS,
        hst.TAG_SEMANTIC_KIND,
        hst.TAG_SEMANTIC_PK,
        hst.TAG_SENSITIVITY,
    }
)


def _validate_tag_keys(tags: Mapping[str, TagValue]) -> None:
    unknown = [key for key in tags if key not in _ALLOWED_TAG_KEYS]
    if unknown:
        msg = f"Unknown tag keys: {', '.join(sorted(unknown))}"
        raise ValueError(msg)


_TAG_DECORATOR_FACTORY = cast("_TagDecoratorFactory", h_tag)


def schema_output(
    *fields: tuple[str, str],
    target_: str | None = None,
) -> Decorator[P, R]:
    """Attach schema metadata to a node (hint-only).

    Returns
    -------
    Decorator[P, R]
        Decorator applying schema metadata.
    """
    return cast("Decorator[P, R]", h_schema.output(*fields, target_=target_))


def tag(**tags: TagValue) -> Decorator[P, R]:
    """Attach canonical tags to a node.

    Returns
    -------
    Decorator[P, R]
        Decorator applying canonical tags.
    """
    return _tag_impl(tags=tags, target_=None)


def tag_output(
    *,
    target_: str | Collection[str] | None = None,
    **tags: TagValue,
) -> Decorator[P, R]:
    """Attach canonical tags to an output node.

    Returns
    -------
    Decorator[P, R]
        Decorator applying canonical tags for output nodes.
    """
    return _tag_impl(tags=tags, target_=target_)


def _tag_impl(
    *,
    tags: Mapping[str, TagValue],
    target_: str | Collection[str] | None,
) -> Decorator[P, R]:
    tags_dict = dict(tags)
    _validate_tag_keys(tags_dict)
    if target_ is None:
        return cast(
            "Decorator[P, R]",
            _TAG_DECORATOR_FACTORY(**tags_dict),
        )
    return cast(
        "Decorator[P, R]",
        _TAG_DECORATOR_FACTORY(**tags_dict, target_=target_),
    )


__all__ = [
    "schema_output",
    "tag",
    "tag_output",
]
