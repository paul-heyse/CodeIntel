"""Dependency helpers for deterministic view materialization.

View builders frequently reference other views (e.g., docs views composed from
analytics views). DuckDB requires referenced views to exist when a view is
materialized, so view materialization must be dependency-aware rather than
relying only on a name-sorted order.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

from sqlglot import exp, parse_one

from codeintel.storage.constants import DUCKDB_DIALECT

__all__ = [
    "build_dependency_graph_from_sql",
    "extract_referenced_table_keys",
    "toposort",
]

if TYPE_CHECKING:
    from collections.abc import Iterable


def extract_referenced_table_keys(sql: str) -> set[str]:
    """Extract referenced table keys from a DuckDB SQL string.

    Parameters
    ----------
    sql
        DuckDB SQL string.

    Returns
    -------
    set[str]
        Set of referenced tables/views, using ``schema.table`` where available.
    """
    root = parse_one(sql, dialect=DUCKDB_DIALECT)
    referenced: set[str] = set()

    for table in root.find_all(exp.Table):
        name = table.name
        schema = table.db
        if schema:
            referenced.add(f"{schema}.{name}".lower())
        else:
            referenced.add(name.lower())

    return referenced


def build_dependency_graph_from_sql(
    view_sql: dict[str, str],
    *,
    view_keys: Iterable[str] | None = None,
) -> dict[str, frozenset[str]]:
    """Build a view dependency mapping from compiled SQL.

    Parameters
    ----------
    view_sql
        Mapping of view table_key to compiled SQL defining that view.
    view_keys
        Optional iterable of view keys considered "nodes" in the graph.
        When omitted, uses ``view_sql.keys()``.

    Returns
    -------
    dict[str, frozenset[str]]
        Mapping of view_key -> dependent view_keys (restricted to the node set).
    """
    nodes = {key.lower() for key in (view_keys or view_sql.keys())}
    deps: dict[str, frozenset[str]] = {}

    for raw_key, sql in view_sql.items():
        key = raw_key.lower()
        referenced = extract_referenced_table_keys(sql)
        deps[key] = frozenset((referenced & nodes) - {key})

    return deps


def toposort(
    nodes: Iterable[str],
    deps: dict[str, frozenset[str]],
    *,
    raise_on_cycle: bool = False,
) -> tuple[str, ...]:
    """Topologically sort nodes given an adjacency mapping.

    Parameters
    ----------
    nodes
        Node identifiers to order.
    deps
        Mapping of node -> direct dependencies (edges node -> dep).
    raise_on_cycle
        When True, raise a ValueError if the dependency graph contains a cycle.

    Returns
    -------
    tuple[str, ...]
        Nodes in topological order. When the graph contains a cycle, returns the
        deterministic name-sorted order as a safe fallback.

    Raises
    ------
    ValueError
        If ``raise_on_cycle`` is True and the graph contains a cycle.
    """
    node_set = {n.lower() for n in nodes}
    outgoing: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = dict.fromkeys(node_set, 0)

    for node in node_set:
        for dep in deps.get(node, frozenset()):
            if dep not in node_set:
                continue
            outgoing[dep].add(node)
            indegree[node] += 1

    ready: deque[str] = deque(sorted([n for n, d in indegree.items() if d == 0]))
    ordered: list[str] = []

    while ready:
        current = ready.popleft()
        ordered.append(current)
        for downstream in sorted(outgoing.get(current, set())):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                ready.append(downstream)

    if len(ordered) != len(node_set):
        if raise_on_cycle:
            cyclic = sorted([n for n, d in indegree.items() if d > 0])
            msg = f"Cycle detected in view dependency graph: {cyclic}"
            raise ValueError(msg)
        return tuple(sorted(node_set))
    return tuple(ordered)
