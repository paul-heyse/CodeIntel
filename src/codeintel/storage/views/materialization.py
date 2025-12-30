"""View compilation and materialization orchestration.

DuckDB requires dependent views to exist when a view is materialized. This
module owns deterministic orchestration over:

- discovering view builders (Hamilton tags)
- compiling SQLGlot expressions to DuckDB SQL
- dependency-aware ordering (CTE-safe via SQLGlot)
- materializing views via DuckDB SQL execution
- syncing derived lineage edges when snapshot identity is present
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import duckdb
from sqlglot.errors import ParseError

from codeintel.storage.datasets import DatasetRegistry
from codeintel.storage.helpers.table_key import is_valid_table_key, split_table_key
from codeintel.storage.metadata.sync import (
    sync_derived_lineage_columns,
    sync_derived_lineage_edges,
)
from codeintel.storage.queries.safe import SqlIngressPolicy, assert_select_perimeter
from codeintel.storage.sqlglot_tools import extract_column_lineage_duckdb
from codeintel.storage.views.dependencies import (
    build_dependency_graph_from_sql,
    extract_referenced_table_keys,
    toposort,
)
from codeintel.storage.views.discovery import discover_view_builders

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hamilton.driver import Driver

    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway.protocol import MinimalGateway

__all__ = ["ViewMaterializationOptions", "materialize_registered_views"]

log = logging.getLogger(__name__)


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
    """Compile and materialize tagged SQLGlot views.

    Parameters
    ----------
    gateway
        Gateway providing DuckDB connection access.
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
    sql_by_view = _compile_view_definitions(
        gateway,
        modules=modules,
        strict=active.strict,
        dr=active.dr,
        tag_query=active.tag_query,
    )
    if not sql_by_view:
        return {}

    _ensure_dependency_tables(
        gateway,
        sql_by_view=sql_by_view,
        strict=active.strict,
    )
    _ensure_view_schemas(
        gateway,
        sql_by_view=sql_by_view,
        strict=active.strict,
    )
    _materialize_views(
        gateway,
        sql_by_view=sql_by_view,
        overwrite=active.overwrite,
        strict=active.strict,
    )
    _sync_view_lineage(gateway, sql_by_view=sql_by_view)
    return dict(sql_by_view)


def _ensure_dependency_tables(
    gateway: MinimalGateway,
    *,
    sql_by_view: dict[str, str],
    strict: bool,
) -> None:
    """Ensure referenced base tables exist before materializing views."""
    candidates = _dependency_candidates(sql_by_view)
    if not candidates:
        return

    dataset_map = _dataset_contract_map(gateway)
    for table_key in candidates:
        if not _should_ensure_dependency(table_key, dataset_map):
            continue
        _ensure_table_dependency(gateway, table_key=table_key, strict=strict)


def _ensure_view_schemas(
    gateway: MinimalGateway,
    *,
    sql_by_view: dict[str, str],
    strict: bool,
) -> None:
    schemas: set[str] = set()
    for view_key in sql_by_view:
        if not is_valid_table_key(view_key):
            log.debug("Skipping schema creation for invalid view key: %s", view_key)
            continue
        schema_name, _ = split_table_key(view_key)
        schemas.add(schema_name)
    for schema_name in sorted(schemas):
        try:
            gateway.policy.create_schema_if_not_exists(schema_name)
        except (duckdb.Error, RuntimeError, ValueError):
            log.exception("Failed to ensure view schema: %s", schema_name)
            if strict:
                raise


def _dependency_candidates(sql_by_view: Mapping[str, str]) -> tuple[str, ...]:
    view_keys = {key.lower() for key in sql_by_view}
    referenced: set[str] = set()
    for sql in sql_by_view.values():
        referenced.update(extract_referenced_table_keys(sql))
    return tuple(sorted(referenced - view_keys))


def _dataset_contract_map(
    gateway: MinimalGateway,
) -> Mapping[str, DatasetContract] | None:
    datasets = getattr(gateway, "datasets", None)
    if isinstance(datasets, DatasetRegistry):
        return datasets.by_table_key
    return None


def _should_ensure_dependency(
    table_key: str,
    dataset_map: Mapping[str, DatasetContract] | None,
) -> bool:
    if not is_valid_table_key(table_key):
        log.debug("Skipping unqualified view dependency: %s", table_key)
        return False
    if dataset_map is None:
        return True
    contract = dataset_map.get(table_key)
    if contract is None:
        log.debug("Skipping unknown view dependency: %s", table_key)
        return False
    return not contract.is_view


def _ensure_table_dependency(
    gateway: MinimalGateway,
    *,
    table_key: str,
    strict: bool,
) -> None:
    try:
        gateway.policy.ensure_table(table_key, create_if_missing=True)
    except (duckdb.Error, KeyError, RuntimeError, ValueError):
        log.exception("Failed to ensure view dependency table: %s", table_key)
        if strict:
            raise


def _compile_view_definitions(
    gateway: MinimalGateway,
    *,
    modules: tuple[ModuleType, ...],
    strict: bool,
    dr: Driver | None,
    tag_query: TagQuery | None,
) -> dict[str, str]:
    _ = gateway
    builders = discover_view_builders(dr=dr, tag_query=tag_query, modules=modules)

    sql_by_view: dict[str, str] = {}
    for spec in builders:
        view_name = spec.table_key
        try:
            expr = spec.builder()
            sql = expr.sql(dialect="duckdb")
            assert_select_perimeter(sql, policy=SqlIngressPolicy())
            sql_by_view[view_name] = sql
        except ParseError as exc:
            log.exception("Failed to parse SQL for view: %s", view_name)
            if strict:
                raise
            log.debug("SQLGlot parse error: %s", exc)
        except (duckdb.Error, KeyError, TypeError, ValueError):
            log.exception("Failed to build view expression: %s", view_name)
            if strict:
                raise
    return sql_by_view


def _materialize_views(
    gateway: MinimalGateway,
    *,
    sql_by_view: dict[str, str],
    overwrite: bool,
    strict: bool,
) -> None:
    deps = build_dependency_graph_from_sql(sql_by_view)
    order_lower = toposort(sql_by_view.keys(), deps, raise_on_cycle=strict)
    original_by_lower = {k.lower(): k for k in sql_by_view}

    for view_key_lower in order_lower:
        view_name = original_by_lower[view_key_lower]
        try:
            database, name = split_table_key(view_name)
            replace = "OR REPLACE " if overwrite else ""
            sql = sql_by_view.get(view_name)
            if sql is None:
                continue
            create_sql = f"CREATE {replace}VIEW {database}.{name} AS {sql}"
            gateway.con.execute(create_sql)
            log.debug("Materialized view: %s", view_name)
        except (duckdb.Error, KeyError, TypeError, ValueError):
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
