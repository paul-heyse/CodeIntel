"""Graph metadata helpers for rustworkx serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from codeintel.build.graphs.rx.payloads import NODE_PAYLOAD_VERSION


@dataclass(frozen=True, slots=True)
class GraphMetadata:
    """Metadata stored in rustworkx graph attributes."""

    cache_version: str
    engine: str
    graph_kind: str
    weight_policy: str
    node_payload_version: str = NODE_PAYLOAD_VERSION

    def as_attrs(self) -> dict[str, object]:
        """Return metadata as a JSON-compatible attribute mapping.

        Returns
        -------
        dict[str, object]
            JSON-compatible metadata mapping.
        """
        return {
            "cache_version": self.cache_version,
            "engine": self.engine,
            "graph_kind": self.graph_kind,
            "weight_policy": self.weight_policy,
            "node_payload_version": self.node_payload_version,
        }

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
        cache_version = _get_str(attrs, "cache_version")
        engine = _get_str(attrs, "engine")
        graph_kind = _get_str(attrs, "graph_kind")
        weight_policy = _get_str(attrs, "weight_policy")
        node_payload_version = _get_str(attrs, "node_payload_version") or NODE_PAYLOAD_VERSION
        if cache_version is None or engine is None or graph_kind is None or weight_policy is None:
            return None
        return cls(
            cache_version=cache_version,
            engine=engine,
            graph_kind=graph_kind,
            weight_policy=weight_policy,
            node_payload_version=node_payload_version,
        )


def _get_str(attrs: dict[str, object], key: str) -> str | None:
    value = attrs.get(key)
    if value is None:
        return None
    return str(value)


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
    graph.attrs = metadata.as_attrs()


__all__ = ["GraphAttrs", "GraphMetadata", "apply_graph_metadata", "metadata_from_graph"]
