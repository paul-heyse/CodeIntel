"""View compilation and materialization orchestration.

DuckDB requires dependent views to exist when a view is materialized. This
module owns deterministic orchestration over:

- discovering view builders (Hamilton tags)
- compiling Ibis expressions to DuckDB SQL
- dependency-aware ordering (CTE-safe via SQLGlot)
- materializing views via the Ibis DuckDB backend
- syncing derived lineage edges when snapshot identity is present
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import TYPE_CHECKING

from ibis.common.exceptions import IbisError

from codeintel.storage.gateway.protocol import DuckDBError, MinimalGateway
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata.sync import sync_derived_lineage_edges
from codeintel.storage.views.dependencies import (
    build_dependency_graph_from_sql,
    extract_referenced_table_keys,
    toposort,
)
from codeintel.storage.views.discovery import discover_view_builders

if TYPE_CHECKING:
    import ibis.expr.types as it

__all__ = ["materialize_registered_views"]

log = logging.getLogger(__name__)


def materialize_registered_views(
    gateway: MinimalGateway,
    *,
    modules: tuple[ModuleType, ...],
    overwrite: bool = True,
    strict: bool = False,
) -> dict[str, str]:
    """Compile and materialize tagged Ibis views.

    Parameters
    ----------
    gateway
        Gateway providing DuckDB connection access and Ibis integration.
    modules
        Python modules containing view builder functions decorated/tagged for discovery.
    overwrite
        When True, overwrite existing views.
    strict
        When True, raise on any view build/materialization failure. When False,
        failures are logged and processing continues.

    Returns
    -------
    dict[str, str]
        Mapping of view table_key -> compiled SQL used for dependency resolution.
    """
    expr_by_view, sql_by_view = _compile_view_definitions(gateway, modules=modules, strict=strict)
    if not sql_by_view:
        return {}

    _materialize_views(
        gateway,
        expr_by_view=expr_by_view,
        sql_by_view=sql_by_view,
        overwrite=overwrite,
        strict=strict,
    )
    _sync_view_lineage(gateway, sql_by_view=sql_by_view)
    return dict(sql_by_view)


def _compile_view_definitions(
    gateway: MinimalGateway,
    *,
    modules: tuple[ModuleType, ...],
    strict: bool,
) -> tuple[dict[str, it.Table], dict[str, str]]:
    ibis_gateway = gateway.ibis
    builders = discover_view_builders(modules=modules)

    expr_by_view: dict[str, it.Table] = {}
    sql_by_view: dict[str, str] = {}
    for spec in builders:
        view_name = spec.table_key
        try:
            expr = spec.builder(ibis_gateway)
            expr_by_view[view_name] = expr
            sql_by_view[view_name] = ibis_gateway.con.compile(expr)
        except (DuckDBError, IbisError, KeyError, TypeError, ValueError):
            log.exception("Failed to build view expression: %s", view_name)
            if strict:
                raise
    return expr_by_view, sql_by_view


def _materialize_views(
    gateway: MinimalGateway,
    *,
    expr_by_view: dict[str, it.Table],
    sql_by_view: dict[str, str],
    overwrite: bool,
    strict: bool,
) -> None:
    ibis_gateway = gateway.ibis
    deps = build_dependency_graph_from_sql(sql_by_view)
    order_lower = toposort(sql_by_view.keys(), deps, raise_on_cycle=strict)
    original_by_lower = {k.lower(): k for k in sql_by_view}

    for view_key_lower in order_lower:
        view_name = original_by_lower[view_key_lower]
        expr = expr_by_view.get(view_name)
        if expr is None:
            continue
        try:
            database, name = split_table_key(view_name)
            ibis_gateway.con.create_view(name, expr, database=database, overwrite=overwrite)
            log.debug("Created view: %s", view_name)
        except (DuckDBError, IbisError, KeyError, TypeError, ValueError):
            log.exception("Failed to create view: %s", view_name)
            if strict:
                raise


def _sync_view_lineage(gateway: MinimalGateway, *, sql_by_view: dict[str, str]) -> None:
    config = getattr(gateway, "config", None)
    repo = getattr(config, "repo", None)
    commit = getattr(config, "commit", None)
    if not (isinstance(repo, str) and repo and isinstance(commit, str) and commit):
        return

    lineage: dict[str, frozenset[str]] = {}
    for raw_key, sql in sql_by_view.items():
        view_key = raw_key.lower()
        lineage[view_key] = frozenset(extract_referenced_table_keys(sql) - {view_key})

    try:
        sync_derived_lineage_edges(gateway.con, repo=repo, commit=commit, lineage=lineage)
    except DuckDBError:
        log.exception("Failed to sync derived lineage edges repo=%s commit=%s", repo, commit)
