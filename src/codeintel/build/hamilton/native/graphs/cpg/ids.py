"""CPG ID helpers."""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id


def stable_cpg_id(table_key: str, pk: Mapping[str, object]) -> int:
    """Public wrapper for stable CPG node IDs.

    Returns
    -------
    int
        Stable CPG node identifier.
    """
    return cpg_node_id(table_key, pk)


__all__ = [
    "stable_cpg_id",
]
