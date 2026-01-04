"""Inspect overlay exports for CPG."""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.cpg import _legacy

inspect_to_ast_edges_to_cpg = _legacy.inspect_to_ast_edges_to_cpg
py_inspect_unwrap_edges_to_cpg = _legacy.py_inspect_unwrap_edges_to_cpg

__all__ = [
    "inspect_to_ast_edges_to_cpg",
    "py_inspect_unwrap_edges_to_cpg",
]
