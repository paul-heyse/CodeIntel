"""Compile DataSaver registries from Hamilton runtime tags."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime


def compile_write_registry(runtime: HamiltonRuntime) -> dict[str, list[dict[str, Any]]]:
    """Group DataSaver nodes by sink with minimal metadata.

    Parameters
    ----------
    runtime
        Hamilton runtime containing a configured Driver.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Mapping of sink name to node metadata entries.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for node in runtime.dr.graph.nodes.values():
        tags = node.tags
        if not isinstance(tags, dict):
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue

        sink = tags.get("hamilton.data_saver.sink")
        sink_name = sink if isinstance(sink, str) and sink else "unknown"
        grouped[sink_name].append(
            {
                "node": node.name,
                "domain": tags.get(ht.TAG_DOMAIN),
                "target": tags.get(ht.TAG_TARGET),
                "table_key": tags.get(ht.TAG_TABLE_KEY),
                "artifact": tags.get(ht.TAG_ARTIFACT),
            }
        )

    for sink in list(grouped):
        grouped[sink] = sorted(grouped[sink], key=lambda row: str(row["node"]))

    return dict(grouped)


def duckdb_materializations(runtime: HamiltonRuntime) -> list[dict[str, Any]]:
    """Return all DuckDB materializations from the runtime.

    Returns
    -------
    list[dict[str, Any]]
        Materialization entries for DuckDB sinks.
    """
    registry = compile_write_registry(runtime)
    duckdb_sinks = {"codeintel.duckdb_rows", "codeintel.duckdb_table"}
    out: list[dict[str, Any]] = []
    for sink in duckdb_sinks:
        out.extend(registry.get(sink, []))
    return out


def artifact_writes(runtime: HamiltonRuntime) -> list[dict[str, Any]]:
    """Return all artifact writes from the runtime.

    Returns
    -------
    list[dict[str, Any]]
        Materialization entries for artifact writes.
    """
    registry = compile_write_registry(runtime)
    return registry.get("codeintel.file_artifact", [])


__all__ = [
    "artifact_writes",
    "compile_write_registry",
    "duckdb_materializations",
]
