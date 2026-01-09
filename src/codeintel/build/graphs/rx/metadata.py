"""Graph metadata helpers for rustworkx serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from codeintel.build.graphs.rx.payloads import EDGE_PAYLOAD_VERSION, NODE_PAYLOAD_VERSION

DEFAULT_GRAPH_CACHE_VERSION = "unknown"
DEFAULT_GRAPH_ENGINE = "rustworkx"
DEFAULT_GRAPH_KIND = "unknown"
DEFAULT_GRAPH_DETERMINISM_TIER = "stable_set"
DEFAULT_GRAPH_EDGE_PAYLOAD_VERSION = EDGE_PAYLOAD_VERSION


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
        attrs: dict[str, object] = {
            "cache_version": self.cache_version,
            "engine": self.engine,
            "graph_kind": self.graph_kind,
            "weight_policy": self.weight_policy,
            "node_payload_version": self.node_payload_version,
            "edge_payload_version": self.edge_payload_version,
            "determinism_tier": self.determinism_tier,
        }
        if self.scan_profile is not None:
            attrs["scan_profile"] = self.scan_profile
        if self.runtime_profile is not None:
            attrs["runtime_profile"] = self.runtime_profile
        if self.ordering_keys is not None:
            attrs["ordering_keys"] = list(self.ordering_keys)
        if self.tie_breaker_keys is not None:
            attrs["tie_breaker_keys"] = list(self.tie_breaker_keys)
        if self.repo is not None:
            attrs["repo"] = self.repo
        if self.commit is not None:
            attrs["commit"] = self.commit
        if self.run_id is not None:
            attrs["run_id"] = self.run_id
        if self.build_timestamp is not None:
            attrs["build_timestamp"] = self.build_timestamp
        if self.dataset_root is not None:
            attrs["dataset_root"] = self.dataset_root
        if self.source_tables:
            attrs["source_tables"] = list(self.source_tables)
        if self.weight_semantics is not None:
            attrs["weight_semantics"] = self.weight_semantics
        if self.is_directed is not None:
            attrs["is_directed"] = self.is_directed
        if self.is_multigraph is not None:
            attrs["is_multigraph"] = self.is_multigraph
        if self.node_count is not None:
            attrs["node_count"] = self.node_count
        if self.edge_count is not None:
            attrs["edge_count"] = self.edge_count
        if self.density is not None:
            attrs["density"] = self.density
        if self.component_count is not None:
            attrs["component_count"] = self.component_count
        if self.scc_count is not None:
            attrs["scc_count"] = self.scc_count
        if self.has_cycles is not None:
            attrs["has_cycles"] = self.has_cycles
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
        edge_payload_version = (
            _get_str(attrs, "edge_payload_version") or DEFAULT_GRAPH_EDGE_PAYLOAD_VERSION
        )
        determinism_tier = _get_str(attrs, "determinism_tier") or DEFAULT_GRAPH_DETERMINISM_TIER
        scan_profile = _get_str(attrs, "scan_profile")
        runtime_profile = _get_str(attrs, "runtime_profile")
        ordering_keys = _get_ordering_keys(attrs.get("ordering_keys"))
        tie_breaker_keys = _get_ordering_keys(attrs.get("tie_breaker_keys"))
        repo = _get_str(attrs, "repo")
        commit = _get_str(attrs, "commit")
        run_id = _get_str(attrs, "run_id")
        build_timestamp = _get_str(attrs, "build_timestamp")
        dataset_root = _get_str(attrs, "dataset_root")
        source_tables = _get_str_list(attrs.get("source_tables"))
        weight_semantics = _get_str(attrs, "weight_semantics")
        is_directed = _get_bool(attrs.get("is_directed"))
        is_multigraph = _get_bool(attrs.get("is_multigraph"))
        node_count = _get_int(attrs.get("node_count"))
        edge_count = _get_int(attrs.get("edge_count"))
        density = _get_float(attrs.get("density"))
        component_count = _get_int(attrs.get("component_count"))
        scc_count = _get_int(attrs.get("scc_count"))
        has_cycles = _get_bool(attrs.get("has_cycles"))
        if weight_policy is None:
            return None
        return cls(
            weight_policy=weight_policy,
            cache_version=cache_version,
            engine=engine,
            graph_kind=graph_kind,
            node_payload_version=node_payload_version,
            edge_payload_version=edge_payload_version,
            determinism_tier=determinism_tier,
            scan_profile=scan_profile,
            runtime_profile=runtime_profile,
            ordering_keys=ordering_keys,
            tie_breaker_keys=tie_breaker_keys,
            repo=repo,
            commit=commit,
            run_id=run_id,
            build_timestamp=build_timestamp,
            dataset_root=dataset_root,
            source_tables=source_tables,
            weight_semantics=weight_semantics,
            is_directed=is_directed,
            is_multigraph=is_multigraph,
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            component_count=component_count,
            scc_count=scc_count,
            has_cycles=has_cycles,
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
        edge_payload_version=metadata.edge_payload_version,
        determinism_tier=metadata.determinism_tier,
        scan_profile=metadata.scan_profile,
        runtime_profile=metadata.runtime_profile,
        ordering_keys=metadata.ordering_keys,
        tie_breaker_keys=metadata.tie_breaker_keys,
        repo=metadata.repo,
        commit=metadata.commit,
        run_id=metadata.run_id,
        build_timestamp=metadata.build_timestamp,
        dataset_root=metadata.dataset_root,
        source_tables=metadata.source_tables,
        weight_semantics=metadata.weight_semantics,
        is_directed=metadata.is_directed,
        is_multigraph=metadata.is_multigraph,
        node_count=metadata.node_count,
        edge_count=metadata.edge_count,
        density=metadata.density,
        component_count=metadata.component_count,
        scc_count=metadata.scc_count,
        has_cycles=metadata.has_cycles,
    )


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
