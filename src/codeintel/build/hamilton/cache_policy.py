"""Central cache behavior policy for Hamilton build runs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from hamilton.caching.adapter import CachingBehavior

from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from hamilton.node import Node

log = logging.getLogger(__name__)

CACHE_BEHAVIOR_TAG = "ci.cache_behavior"
CACHE_SALT_TAG = "ci.cache_salt"
CACHE_SALT_DEP_NAME = "__cache_salt__"

_CACHE_BEHAVIOR_VALUES: dict[str, CachingBehavior] = {
    "default": CachingBehavior.DEFAULT,
    "recompute": CachingBehavior.RECOMPUTE,
    "ignore": CachingBehavior.IGNORE,
    "disable": CachingBehavior.DISABLE,
}

_NON_CACHEABLE_NODE_NAMES: frozenset[str] = frozenset(
    {
        "cache_index",
        "cache_key_resolver",
        "catalog",
        "env",
        "plan_context",
        "plan_request",
        "runtime_fingerprint",
        "schema_index",
        "semantic_registry",
        "tag_query",
    }
)


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Cache behavior resolution rules."""

    default_behavior: CachingBehavior = CachingBehavior.DEFAULT
    node_type_overrides: Mapping[str, CachingBehavior] = field(default_factory=dict)
    name_overrides: Mapping[str, CachingBehavior] = field(default_factory=dict)
    prefix_overrides: Mapping[str, CachingBehavior] = field(default_factory=dict)
    suffix_overrides: Mapping[str, CachingBehavior] = field(default_factory=dict)
    behavior_tag: str = CACHE_BEHAVIOR_TAG
    salt_tag: str = CACHE_SALT_TAG


def default_cache_policy() -> CachePolicy:
    """Return the default cache behavior policy for build runs.

    Returns
    -------
    CachePolicy
        Default cache policy for build execution.
    """
    name_overrides = dict.fromkeys(_NON_CACHEABLE_NODE_NAMES, CachingBehavior.DISABLE)
    name_overrides["plan"] = CachingBehavior.DISABLE
    return CachePolicy(
        default_behavior=CachingBehavior.DEFAULT,
        node_type_overrides={
            ht.NODE_TYPE_MATERIALIZE: CachingBehavior.RECOMPUTE,
        },
        name_overrides=name_overrides,
        prefix_overrides={
            "plan_": CachingBehavior.DISABLE,
        },
        suffix_overrides={
            "__finalize_context": CachingBehavior.DISABLE,
        },
    )


def resolve_behavior(
    node: Node,
    policy: CachePolicy,
    *,
    default_behavior: CachingBehavior,
) -> CachingBehavior:
    """Resolve the effective cache behavior for a Hamilton node.

    Parameters
    ----------
    node
        Hamilton node instance.
    policy
        Cache policy describing overrides.
    default_behavior
        Fallback behavior when no policy rule matches.

    Returns
    -------
    CachingBehavior
        Resolved cache behavior for the node.
    """
    tags = _node_tags(node)
    tagged_behavior = _behavior_from_tag(tags.get(policy.behavior_tag))
    if tagged_behavior is not None:
        return tagged_behavior

    node_name = getattr(node, "name", None)
    if isinstance(node_name, str):
        if node_name in policy.name_overrides:
            return policy.name_overrides[node_name]
        for prefix, behavior in policy.prefix_overrides.items():
            if node_name.startswith(prefix):
                return behavior
        for suffix, behavior in policy.suffix_overrides.items():
            if node_name.endswith(suffix):
                return behavior

    node_type = tags.get(ht.TAG_NODE_TYPE)
    if isinstance(node_type, str) and node_type in policy.node_type_overrides:
        return policy.node_type_overrides[node_type]

    return default_behavior


def is_salt_sensitive(node: Node, policy: CachePolicy) -> bool:
    """Return True when cache salt should be applied to the node.

    Parameters
    ----------
    node
        Hamilton node instance.
    policy
        Cache policy describing salt tagging.

    Returns
    -------
    bool
        True when the node should include cache salt in its cache key.
    """
    tags = _node_tags(node)
    return _bool_from_tag(tags.get(policy.salt_tag))


def cache_salt(runtime_fingerprint: str) -> str:
    """Return a stable cache salt value for cache key derivation.

    Parameters
    ----------
    runtime_fingerprint
        Runtime fingerprint for the composed Hamilton graph.

    Returns
    -------
    str
        Cache salt value derived from the runtime fingerprint.
    """
    return f"codeintel:{runtime_fingerprint}"


def _node_tags(node: Node) -> Mapping[str, object]:
    tags = getattr(node, "tags", None)
    if isinstance(tags, Mapping):
        return cast("Mapping[str, object]", tags)
    return {}


def _behavior_from_tag(value: object) -> CachingBehavior | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    behavior = _CACHE_BEHAVIOR_VALUES.get(normalized)
    if behavior is None:
        log.warning("Unknown cache behavior tag %r; ignoring", value)
    return behavior


def _bool_from_tag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "on"}
    return False


__all__ = [
    "CACHE_BEHAVIOR_TAG",
    "CACHE_SALT_DEP_NAME",
    "CACHE_SALT_TAG",
    "CachePolicy",
    "cache_salt",
    "default_cache_policy",
    "is_salt_sensitive",
    "resolve_behavior",
]
