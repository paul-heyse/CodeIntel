"""DEPRECATED: Most graphs adapters have been removed.

.. deprecated:: 4.0.0
    The graphs adapters package has been deprecated. The unused adapter classes
    (DuckDBStorageAdapter, LibCSTParsingAdapter, NxEngineAdapter) have been
    removed as they had no production callers.

    The callgraph persistence utilities have been relocated:

    - For ``dedupe_edge_rows``: Use ``codeintel.graphs.compute.callgraph.persistence``
    - For ``default_edge_key``: Use ``codeintel.graphs.compute.callgraph.persistence``
    - For ``persist_call_graph_edges``: Use ``ctx.write_table()`` in Hamilton plugins
    - For ``persist_call_graph_nodes``: Use ``ctx.write_table()`` in Hamilton plugins

    Storage operations should use ``StorageGateway`` directly.
    Parsing is handled by the types in ``codeintel.graphs.ports.parsing``.
    Graph engines should use ``NxGraphEngine`` directly.

This stub module exists only to provide helpful error messages.
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Provide helpful error messages for deprecated imports.

    Parameters
    ----------
    name
        Name of the attribute being accessed.

    Raises
    ------
    ImportError
        Raised for known deprecated imports with migration guidance.
    AttributeError
        Raised for unknown attributes.
    """
    migration_map = {
        "DuckDBStorageAdapter": "removed - use StorageGateway directly",
        "LibCSTParsingAdapter": "removed - parsing handled by graphs/ports/parsing.py types",
        "NxEngineAdapter": "removed - use NxGraphEngine directly",
        "dedupe_edge_rows": "codeintel.graphs.compute.callgraph.persistence.dedupe_edge_rows",
        "default_edge_key": "codeintel.graphs.compute.callgraph.persistence.default_edge_key",
        "persist_call_graph_edges": "removed - use ctx.write_table() in Hamilton plugins",
        "persist_call_graph_nodes": "removed - use ctx.write_table() in Hamilton plugins",
        "callgraph_persistence": "codeintel.graphs.compute.callgraph.persistence",
    }

    if name in migration_map:
        suggestion = migration_map[name]
        message = (
            f"'{name}' has been removed from codeintel.graphs.adapters. Use {suggestion} instead."
        )
        raise ImportError(message)

    message = f"module 'codeintel.graphs.adapters' has no attribute '{name}'"
    raise AttributeError(message)


__all__: list[str] = []
