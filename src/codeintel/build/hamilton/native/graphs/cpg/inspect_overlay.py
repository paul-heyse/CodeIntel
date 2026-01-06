"""Inspect overlay exports for CPG."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg2.planes.overlays_inspect import (
    cpg2_edges__inspect_to_ast,
    cpg2_edges__py_inspect_unwrap,
)


def py_inspect_unwrap_edges_to_cpg(
    unwrap_hops: pa.Table,
) -> pa.Table:
    """Public wrapper for inspect unwrap edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of unwrap edges.
    """
    return cpg2_edges__py_inspect_unwrap(unwrap_hops)


def inspect_to_ast_edges_to_cpg(
    inspect_objects: pa.Table,
    inspect_source: pa.Table,
    ast_nodes: pa.Table,
) -> pa.Table:
    """Public wrapper for inspect-to-AST anchor edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of inspect anchor edges.
    """
    return cpg2_edges__inspect_to_ast(inspect_objects, inspect_source, ast_nodes)


__all__ = [
    "inspect_to_ast_edges_to_cpg",
    "py_inspect_unwrap_edges_to_cpg",
]
