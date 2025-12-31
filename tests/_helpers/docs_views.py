"""Helpers for docs view tests (indexes, seeding, profiling DB creation)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import duckdb

from codeintel.build.hamilton.native.views.view_outputs import ViewPlan, view_plan_map
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.schema import apply_all_schemas
from codeintel.storage.views.dependencies import toposort

if TYPE_CHECKING:
    from pathlib import Path

    from duckdb import DuckDBPyConnection


def list_indexes(con: DuckDBPyConnection, *, schema: str, table: str) -> set[str]:
    """
    Return index names for a table.

    Returns
    -------
    set[str]
        Index names defined for the table.
    """
    rows = con.execute(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE schema_name = ? AND table_name = ?
        """,
        [schema, table],
    ).fetchall()
    return {str(row[0]) for row in rows}


def seed_subsystem(con: DuckDBPyConnection, *, overrides: dict[str, object] | None = None) -> None:
    """
    Seed analytics.subsystems with a single subsystem row.

    Parameters
    ----------
    con
        Active DuckDB connection.
    overrides
        Optional overrides for the default subsystem payload.
    """
    base: dict[str, object] = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "subsystem_id": "subsysdemo",
        "name": "Subsystem Demo",
        "description": "demo subsystem",
        "module_count": 1,
        "function_count": 1,
    }
    if overrides:
        base.update(overrides)
    con.execute(
        """
        INSERT OR REPLACE INTO analytics.subsystems (
            repo, commit, subsystem_id, name, description,
            module_count, modules_json, entrypoints_json,
            internal_edge_count, external_edge_count, fan_in, fan_out,
            function_count, avg_risk_score, max_risk_score,
            high_risk_function_count, risk_level, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            base["repo"],
            base["commit"],
            base["subsystem_id"],
            base["name"],
            base["description"],
            base["module_count"],
            0,
            0,
            0,
            0,
            base["function_count"],
            0.1,
            0.2,
            0,
            "low",
        ],
    )


def _normalize_view_keys(
    view_map: Mapping[str, ViewPlan],
    view_keys: Iterable[str] | None,
) -> set[str]:
    if view_keys is None:
        return set(view_map)
    lower_map = {key.lower(): key for key in view_map}
    pending = list(view_keys)
    selected: set[str] = set()
    missing: set[str] = set()
    while pending:
        raw = pending.pop()
        resolved = lower_map.get(raw.lower())
        if resolved is None:
            missing.add(raw)
            continue
        if resolved in selected:
            continue
        selected.add(resolved)
        pending.extend(dep for dep in view_map[resolved].dependencies if dep.lower() in lower_map)
    if missing:
        msg = f"Unknown view keys requested: {sorted(missing)}"
        raise KeyError(msg)
    return selected


def _view_dependency_graph(
    view_map: Mapping[str, ViewPlan],
    *,
    view_keys: set[str],
) -> dict[str, frozenset[str]]:
    selected_lower = {key.lower() for key in view_keys}
    deps: dict[str, frozenset[str]] = {}
    for view_key in view_keys:
        ref_set = {
            dep.lower() for dep in view_map[view_key].dependencies if dep.lower() in selected_lower
        }
        deps[view_key.lower()] = frozenset(ref_set - {view_key.lower()})
    return deps


def materialize_view_plans(
    con: DuckDBPyConnection,
    *,
    view_keys: Iterable[str] | None = None,
) -> None:
    """Materialize precompiled view plans into DuckDB (test helper).

    Parameters
    ----------
    con
        DuckDB connection to receive the views.
    view_keys
        Optional view keys to materialize; dependencies are added automatically.
        When None, all view plans are materialized.
    """
    view_map = view_plan_map()
    selected = _normalize_view_keys(view_map, view_keys)
    if not selected:
        return
    deps = _view_dependency_graph(view_map, view_keys=selected)
    order = toposort([key.lower() for key in selected], deps, raise_on_cycle=True)
    original_by_lower = {key.lower(): key for key in selected}
    for view_key_lower in order:
        view_key = original_by_lower[view_key_lower]
        spec = view_map[view_key]
        schema, table = split_table_key(view_key)
        con.execute(f"CREATE OR REPLACE VIEW {schema}.{table} AS {spec.sql}")


def create_bootstrapped_docs_db(db_path: Path) -> None:
    """Create a file-backed DuckDB with schemas, views, and metadata bootstrapped."""
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"ATTACH DATABASE '{db_path}' AS test_db (STORAGE_VERSION 'v1.4.0')")
        con.execute("USE test_db")
        apply_all_schemas(con)
        materialize_view_plans(con)
        bootstrap_metadata_datasets(con, include_views=True)
    finally:
        con.close()


__all__ = [
    "create_bootstrapped_docs_db",
    "list_indexes",
    "materialize_view_plans",
    "seed_subsystem",
]
