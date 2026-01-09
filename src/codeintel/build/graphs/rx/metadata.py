"""Graph metadata helpers for rustworkx serialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, TypedDict

from codeintel.build.graphs.rx.payloads import EDGE_PAYLOAD_VERSION, NODE_PAYLOAD_VERSION

DEFAULT_GRAPH_CACHE_VERSION = "unknown"
DEFAULT_GRAPH_ENGINE = "rustworkx"
DEFAULT_GRAPH_KIND = "unknown"
DEFAULT_GRAPH_DETERMINISM_TIER = "stable_set"
DEFAULT_GRAPH_EDGE_PAYLOAD_VERSION = EDGE_PAYLOAD_VERSION


class _GraphMetadataArgs(TypedDict, total=False):
    weight_policy: str
    cache_version: str
    engine: str
    graph_kind: str
    node_payload_version: str
    edge_payload_version: str
    determinism_tier: str
    scan_profile: str | None
    runtime_profile: str | None
    ordering_keys: tuple[str, ...] | None
    tie_breaker_keys: tuple[str, ...] | None
    repo: str | None
    commit: str | None
    run_id: str | None
    build_timestamp: str | None
    dataset_root: str | None
    source_tables: tuple[str, ...]
    weight_semantics: str | None
    is_directed: bool | None
    is_multigraph: bool | None
    node_count: int | None
    edge_count: int | None
    density: float | None
    component_count: int | None
    scc_count: int | None
    has_cycles: bool | None


@dataclass(frozen=True, slots=True)
class GraphMetadata:
    """Metadata stored in rustworkx graph attributes."""

    weight_policy: str
    cache_version: str = DEFAULT_GRAPH_CACHE_VERSION
    engine: str = DEFAULT_GRAPH_ENGINE
    graph_kind: str = DEFAULT_GRAPH_KIND
    node_payload_version: str = NODE_PAYLOAD_VERSION
    edge_payload_version: str = DEFAULT_GRAPH_EDGE_PAYLOAD_VERSION
    determinism_tier: str = DEFAULT_GRAPH_DETERMINISM_TIER
    scan_profile: str | None = None
    runtime_profile: str | None = None
    ordering_keys: tuple[str, ...] | None = None
    tie_breaker_keys: tuple[str, ...] | None = None
    repo: str | None = None
    commit: str | None = None
    run_id: str | None = None
    build_timestamp: str | None = None
    dataset_root: str | None = None
    source_tables: tuple[str, ...] = ()
    weight_semantics: str | None = None
    is_directed: bool | None = None
    is_multigraph: bool | None = None
    node_count: int | None = None
    edge_count: int | None = None
    density: float | None = None
    component_count: int | None = None
    scc_count: int | None = None
    has_cycles: bool | None = None

    def as_attrs(self) -> dict[str, object]:
        """Return metadata as a JSON-compatible attribute mapping.

        Returns
        -------
        dict[str, object]
            JSON-compatible metadata mapping.
        """
        attrs = _base_attrs(self)
        attrs.update(_optional_attrs(self))
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
        weight_policy = _get_str(attrs, "weight_policy")
        if weight_policy is None:
            return None
        fields = _parsed_metadata_fields(attrs)
        fields["weight_policy"] = weight_policy
        return cls(**fields)


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
    return replace(metadata, weight_policy=weight_policy)


def _base_attrs(metadata: GraphMetadata) -> dict[str, object]:
    return {
        "cache_version": metadata.cache_version,
        "engine": metadata.engine,
        "graph_kind": metadata.graph_kind,
        "weight_policy": metadata.weight_policy,
        "node_payload_version": metadata.node_payload_version,
        "edge_payload_version": metadata.edge_payload_version,
        "determinism_tier": metadata.determinism_tier,
    }


def _optional_attrs(metadata: GraphMetadata) -> dict[str, object]:
    attrs: dict[str, object] = {}
    _set_optional(attrs, "scan_profile", metadata.scan_profile)
    _set_optional(attrs, "runtime_profile", metadata.runtime_profile)
    _set_optional(attrs, "ordering_keys", _as_list(metadata.ordering_keys))
    _set_optional(attrs, "tie_breaker_keys", _as_list(metadata.tie_breaker_keys))
    _set_optional(attrs, "repo", metadata.repo)
    _set_optional(attrs, "commit", metadata.commit)
    _set_optional(attrs, "run_id", metadata.run_id)
    _set_optional(attrs, "build_timestamp", metadata.build_timestamp)
    _set_optional(attrs, "dataset_root", metadata.dataset_root)
    if metadata.source_tables:
        attrs["source_tables"] = list(metadata.source_tables)
    _set_optional(attrs, "weight_semantics", metadata.weight_semantics)
    _set_optional(attrs, "is_directed", metadata.is_directed)
    _set_optional(attrs, "is_multigraph", metadata.is_multigraph)
    _set_optional(attrs, "node_count", metadata.node_count)
    _set_optional(attrs, "edge_count", metadata.edge_count)
    _set_optional(attrs, "density", metadata.density)
    _set_optional(attrs, "component_count", metadata.component_count)
    _set_optional(attrs, "scc_count", metadata.scc_count)
    _set_optional(attrs, "has_cycles", metadata.has_cycles)
    return attrs


def _parsed_metadata_fields(attrs: dict[str, object]) -> _GraphMetadataArgs:
    return {
        "cache_version": _get_str(attrs, "cache_version") or DEFAULT_GRAPH_CACHE_VERSION,
        "engine": _get_str(attrs, "engine") or DEFAULT_GRAPH_ENGINE,
        "graph_kind": _get_str(attrs, "graph_kind") or DEFAULT_GRAPH_KIND,
        "node_payload_version": _get_str(attrs, "node_payload_version") or NODE_PAYLOAD_VERSION,
        "edge_payload_version": _get_str(attrs, "edge_payload_version")
        or DEFAULT_GRAPH_EDGE_PAYLOAD_VERSION,
        "determinism_tier": _get_str(attrs, "determinism_tier")
        or DEFAULT_GRAPH_DETERMINISM_TIER,
        "scan_profile": _get_str(attrs, "scan_profile"),
        "runtime_profile": _get_str(attrs, "runtime_profile"),
        "ordering_keys": _get_ordering_keys(attrs.get("ordering_keys")),
        "tie_breaker_keys": _get_ordering_keys(attrs.get("tie_breaker_keys")),
        "repo": _get_str(attrs, "repo"),
        "commit": _get_str(attrs, "commit"),
        "run_id": _get_str(attrs, "run_id"),
        "build_timestamp": _get_str(attrs, "build_timestamp"),
        "dataset_root": _get_str(attrs, "dataset_root"),
        "source_tables": _get_str_list(attrs.get("source_tables")),
        "weight_semantics": _get_str(attrs, "weight_semantics"),
        "is_directed": _get_bool(attrs.get("is_directed")),
        "is_multigraph": _get_bool(attrs.get("is_multigraph")),
        "node_count": _get_int(attrs.get("node_count")),
        "edge_count": _get_int(attrs.get("edge_count")),
        "density": _get_float(attrs.get("density")),
        "component_count": _get_int(attrs.get("component_count")),
        "scc_count": _get_int(attrs.get("scc_count")),
        "has_cycles": _get_bool(attrs.get("has_cycles")),
    }


def _set_optional(attrs: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        attrs[key] = value


def _as_list(values: tuple[str, ...] | None) -> list[str] | None:
    if values is None:
        return None
    return list(values)


def _get_str(attrs: dict[str, object], key: str) -> str | None:
    value = attrs.get(key)
    if value is None:
        return None
    return str(value)


def _get_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _get_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _get_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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


def _get_str_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item is not None)
    return ()


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
