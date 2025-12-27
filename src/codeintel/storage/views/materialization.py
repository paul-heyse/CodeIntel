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
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import duckdb
from ibis.common.exceptions import IbisError, TableNotFound

from codeintel.storage.gateway import ibis_facade
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata.sync import (
    sync_derived_lineage_columns,
    sync_derived_lineage_edges,
)
from codeintel.storage.sqlglot_tools import extract_column_lineage_duckdb
from codeintel.storage.views.dependencies import (
    build_dependency_graph_from_sql,
    extract_referenced_table_keys,
    toposort,
)
from codeintel.storage.views.discovery import discover_view_builders

if TYPE_CHECKING:
    import ibis.expr.types as it
    from hamilton.driver import Driver
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.storage.gateway.protocol import MinimalGateway

__all__ = ["ViewMaterializationOptions", "materialize_registered_views"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _IbisGatewayAdapter:
    """Adapter that exposes Ibis backend access for view builders."""

    con: DuckDBBackend

    def table(self, table_name: str) -> it.Table:
        if "." in table_name:
            database, name = split_table_key(table_name)
            return self.con.table(name, database=database)
        return self.con.table(table_name)


@dataclass(frozen=True, slots=True)
class ViewMaterializationOptions:
    """Options controlling view compilation and materialization."""

    overwrite: bool = True
    strict: bool = False
    dr: Driver | None = None
    tag_query: TagQuery | None = None


def materialize_registered_views(
    gateway: MinimalGateway,
    *,
    modules: tuple[ModuleType, ...],
    options: ViewMaterializationOptions | None = None,
) -> dict[str, str]:
    """Compile and materialize tagged Ibis views.

    Parameters
    ----------
    gateway
        Gateway providing DuckDB connection access and Ibis integration.
    modules
        Python modules containing view builder functions decorated/tagged for discovery.
    options
        Optional materialization options (overwrite, strict, driver, tag query).

    Returns
    -------
    dict[str, str]
        Mapping of view table_key -> compiled SQL used for dependency resolution.
    """
    active = options or ViewMaterializationOptions()
    expr_by_view, sql_by_view = _compile_view_definitions(
        gateway,
        modules=modules,
        strict=active.strict,
        dr=active.dr,
        tag_query=active.tag_query,
    )
    if not sql_by_view:
        return {}

    _materialize_views(
        gateway,
        expr_by_view=expr_by_view,
        sql_by_view=sql_by_view,
        overwrite=active.overwrite,
        strict=active.strict,
    )
    _sync_view_lineage(gateway, sql_by_view=sql_by_view)
    return dict(sql_by_view)


def _compile_view_definitions(
    gateway: MinimalGateway,
    *,
    modules: tuple[ModuleType, ...],
    strict: bool,
    dr: Driver | None,
    tag_query: TagQuery | None,
) -> tuple[dict[str, it.Table], dict[str, str]]:
    ibis_backend = ibis_facade.backend(gateway)
    ibis_gateway = _IbisGatewayAdapter(ibis_backend)
    builders = discover_view_builders(dr=dr, tag_query=tag_query, modules=modules)

    expr_by_view: dict[str, it.Table] = {}
    sql_by_view: dict[str, str] = {}
    for spec in builders:
        view_name = spec.table_key
        try:
            expr = spec.builder(ibis_gateway)
            expr_by_view[view_name] = expr
            sql_by_view[view_name] = ibis_gateway.con.compile(expr)
        except TableNotFound:
            log.warning("Skipping view with missing source tables: %s", view_name)
        except duckdb.CatalogException as exc:
            if "does not exist" in str(exc):
                log.warning("Skipping view with missing source tables: %s", view_name)
                continue
            log.exception("Failed to build view expression: %s", view_name)
            if strict:
                raise
        except (duckdb.Error, IbisError, KeyError, TypeError, ValueError):
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
    ibis_backend = ibis_facade.backend(gateway)
    ibis_gateway = _IbisGatewayAdapter(ibis_backend)
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
            log.debug("Materialized view: %s", view_name)
        except (duckdb.Error, IbisError, KeyError, TypeError, ValueError):
            log.exception("Failed to materialize view: %s", view_name)
            if strict:
                raise


def _sync_view_lineage(gateway: MinimalGateway, *, sql_by_view: dict[str, str]) -> None:
    config = getattr(gateway, "config", None)
    repo = getattr(config, "repo", None)
    commit = getattr(config, "commit", None)
    if not (isinstance(repo, str) and repo and isinstance(commit, str) and commit):
        return

    lineage: dict[str, frozenset[str]] = {}
    column_lineage: dict[str, dict[str, frozenset[str]]] = {}
    for raw_key, sql in sql_by_view.items():
        view_key = raw_key.lower()
        lineage[view_key] = frozenset(extract_referenced_table_keys(sql) - {view_key})
        column_lineage[view_key] = extract_column_lineage_duckdb(sql)

    try:
        sync_derived_lineage_edges(gateway.con, repo=repo, commit=commit, lineage=lineage)
    except duckdb.Error:
        log.exception("Failed to sync derived lineage edges repo=%s commit=%s", repo, commit)

    try:
        sync_derived_lineage_columns(
            gateway.con,
            repo=repo,
            commit=commit,
            lineage=column_lineage,
        )
    except duckdb.Error:
        log.exception("Failed to sync derived lineage columns repo=%s commit=%s", repo, commit)
