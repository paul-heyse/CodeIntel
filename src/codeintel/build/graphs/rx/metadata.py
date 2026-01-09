"""Graph metadata helpers for rustworkx serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from codeintel.build.graphs.rx.payloads import NODE_PAYLOAD_VERSION

DEFAULT_GRAPH_CACHE_VERSION = "unknown"
DEFAULT_GRAPH_ENGINE = "rustworkx"
DEFAULT_GRAPH_KIND = "unknown"
DEFAULT_GRAPH_DETERMINISM_TIER = "stable_set"


@dataclass(frozen=True, slots=True)
class GraphMetadata:
    """Metadata stored in rustworkx graph attributes."""

    weight_policy: str
    cache_version: str = DEFAULT_GRAPH_CACHE_VERSION
    engine: str = DEFAULT_GRAPH_ENGINE
    graph_kind: str = DEFAULT_GRAPH_KIND
    node_payload_version: str = NODE_PAYLOAD_VERSION
    determinism_tier: str = DEFAULT_GRAPH_DETERMINISM_TIER
    scan_profile: str | None = None
    ordering_keys: tuple[str, ...] | None = None

    def as_attrs(self) -> dict[str, object]:
        """Return metadata as a JSON-compatible attribute mapping.

        Returns
        -------
        dict[str, object]
            JSON-compatible metadata mapping.
        """
        attrs: dict[str, object] = {
            "cache_version": self.cache_version,
            "engine": self.engine,
            "graph_kind": self.graph_kind,
            "weight_policy": self.weight_policy,
            "node_payload_version": self.node_payload_version,
            "determinism_tier": self.determinism_tier,
        }
        if self.scan_profile is not None:
            attrs["scan_profile"] = self.scan_profile
        if self.ordering_keys is not None:
            attrs["ordering_keys"] = list(self.ordering_keys)
        return attrs

    @classmethod
    def from_attrs(cls, attrs: object) -> GraphMetadata | None:
        """Parse GraphMetadata from a graph attrs payload.

        Returns
        -------
        GraphMetadata | None
            Parsed metadata when the payload is valid.
        """
        if not isinstance(attrs, dict):
            return None
        cache_version = _get_str(attrs, "cache_version") or DEFAULT_GRAPH_CACHE_VERSION
        engine = _get_str(attrs, "engine") or DEFAULT_GRAPH_ENGINE
        graph_kind = _get_str(attrs, "graph_kind") or DEFAULT_GRAPH_KIND
        weight_policy = _get_str(attrs, "weight_policy")
        node_payload_version = _get_str(attrs, "node_payload_version") or NODE_PAYLOAD_VERSION
        determinism_tier = _get_str(attrs, "determinism_tier") or DEFAULT_GRAPH_DETERMINISM_TIER
        scan_profile = _get_str(attrs, "scan_profile")
        ordering_keys = _get_ordering_keys(attrs.get("ordering_keys"))
        if weight_policy is None:
            return None
        return cls(
            weight_policy=weight_policy,
            cache_version=cache_version,
            engine=engine,
            graph_kind=graph_kind,
            node_payload_version=node_payload_version,
            determinism_tier=determinism_tier,
            scan_profile=scan_profile,
            ordering_keys=ordering_keys,
        )


def metadata_with_weight_policy(
    metadata: GraphMetadata | None,
    *,
    weight_policy: str,
) -> GraphMetadata:
    """Return metadata with a required weight policy applied.

    Returns
    -------
    GraphMetadata
        Metadata with a normalized weight policy.
    """
    if metadata is None:
        return GraphMetadata(weight_policy=weight_policy)
    if metadata.weight_policy == weight_policy:
        return metadata
    return GraphMetadata(
        weight_policy=weight_policy,
        cache_version=metadata.cache_version,
        engine=metadata.engine,
        graph_kind=metadata.graph_kind,
        node_payload_version=metadata.node_payload_version,
        determinism_tier=metadata.determinism_tier,
        scan_profile=metadata.scan_profile,
        ordering_keys=metadata.ordering_keys,
    )


def _get_str(attrs: dict[str, object], key: str) -> str | None:
    value = attrs.get(key)
    if value is None:
        return None
    return str(value)


def _get_ordering_keys(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        keys = [str(item) for item in value if item is not None]
        if not keys:
            return None
        return tuple(keys)
    return None


class GraphAttrs(Protocol):
    """Protocol for rustworkx graphs that carry attrs."""

    attrs: dict[str, object]


def metadata_from_graph(graph: GraphAttrs) -> GraphMetadata | None:
    """Extract metadata from a rustworkx graph if present.

    Returns
    -------
    GraphMetadata | None
        Parsed metadata when available on the graph.
    """
    return GraphMetadata.from_attrs(graph.attrs)


def apply_graph_metadata(graph: GraphAttrs, metadata: GraphMetadata) -> None:
    """Attach metadata to a rustworkx graph."""
    current = graph.attrs if isinstance(graph.attrs, dict) else {}
    merged = {**current, **metadata.as_attrs()}
    graph.attrs = merged


__all__ = [
    "GraphAttrs",
    "GraphMetadata",
    "apply_graph_metadata",
    "metadata_from_graph",
    "metadata_with_weight_policy",
]
