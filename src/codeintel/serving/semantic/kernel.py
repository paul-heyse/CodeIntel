"""Semantic query kernel - unified API for HTTP and MCP.

The kernel provides a single entry point for semantic layer queries, used by
both FastAPI routes and MCP tools.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ibis
import pandas as pd

from codeintel.build.spec import buildspec_from_json
from codeintel.serving.search.models import SearchQueryResponse, SearchResult
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.models import SemanticExplainResponse, SemanticQueryResponse
from codeintel.serving.semantic.query_builder import SemanticQueryPlan, build_query
from codeintel.serving.semantic.registry import SemanticRegistry
from codeintel.storage.gateway.minimal import MinimalStorageGateway

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.spec import BuildSpec
    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.models import SemanticQueryRequest, SemanticViewSpec
    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.gateway.protocol import DuckDBConnection

LOG = logging.getLogger(__name__)

_SEARCH_TABLE_SCHEMA = "docs"
_SEARCH_TABLE_NAME = "search_documents"
_SEARCH_TABLE_KEY = "docs.search_documents"
_SEARCH_FTS_SCHEMA = "fts_docs_search_documents"

_SQL_SEARCH_FTS = """
SELECT kind, name, module, rel_path, ref_goid_h128, score
FROM (
    SELECT
        kind,
        name,
        module,
        rel_path,
        ref_goid_h128,
        fts_docs_search_documents.match_bm25(doc_id, ?) AS score
    FROM docs.search_documents
) ranked
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_FTS_KINDS = """
SELECT kind, name, module, rel_path, ref_goid_h128, score
FROM (
    SELECT
        kind,
        name,
        module,
        rel_path,
        ref_goid_h128,
        fts_docs_search_documents.match_bm25(doc_id, ?) AS score
    FROM docs.search_documents
    WHERE kind = ANY(?)
) ranked
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_LIKE = """
SELECT kind, name, module, rel_path, ref_goid_h128, NULL AS score
FROM docs.search_documents
WHERE (
    COALESCE(text, '') ILIKE '%' || ? || '%'
    OR COALESCE(name, '') ILIKE '%' || ? || '%'
    OR COALESCE(module, '') ILIKE '%' || ? || '%'
)
ORDER BY kind, name
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_LIKE_KINDS = """
SELECT kind, name, module, rel_path, ref_goid_h128, NULL AS score
FROM docs.search_documents
WHERE (
    COALESCE(text, '') ILIKE '%' || ? || '%'
    OR COALESCE(name, '') ILIKE '%' || ? || '%'
    OR COALESCE(module, '') ILIKE '%' || ? || '%'
)
AND kind = ANY(?)
ORDER BY kind, name
LIMIT ? OFFSET ?
"""


def _sanitize_float_nan(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _sanitize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: _sanitize_float_nan(v) for k, v in row.items()} for row in rows]


def _format_explain_rows(rows: Sequence[Sequence[object]]) -> str:
    plan_lines: list[str] = []
    for row in rows:
        if not row:
            continue
        plan_lines.append(str(row[1] if len(row) > 1 else row[0]))
    return "\n".join(plan_lines)


@dataclass
class SemanticQueryKernel:
    """Unified query kernel for semantic layer access.

    Parameters
    ----------
    db
        Database manager for connection access.
    """

    db: ServingDBManager
    settings: ServingSettings

    def _load_registry(self) -> SemanticRegistry:
        """Load semantic registry from current snapshot.

        Returns
        -------
        SemanticRegistry
            Loaded semantic registry.
        """
        pointer = self.db.current_pointer()
        return SemanticRegistry.load(pointer.semantic_registry_path)

    def _load_inventory(self) -> SchemaInventory:
        """Load schema inventory from current snapshot.

        Returns
        -------
        SchemaInventory
            Loaded schema inventory.
        """
        pointer = self.db.current_pointer()
        return SchemaInventory.load(pointer.schema_manifest_path)

    def _load_buildspec(self) -> BuildSpec:
        """Load BuildSpec from current snapshot.

        Returns
        -------
        BuildSpec
            Loaded BuildSpec contract.
        """
        pointer = self.db.current_pointer()
        payload = pointer.buildspec_path.read_text(encoding="utf-8")
        return buildspec_from_json(payload)

    def _resolve_allowed_columns(
        self,
        *,
        view: SemanticViewSpec,
        inventory: SchemaInventory,
    ) -> list[str]:
        """Resolve allowed columns for a view, enforcing schema manifest when enabled.

        Parameters
        ----------
        view
            Semantic view specification.
        inventory
            Schema inventory loaded from the current snapshot.

        Returns
        -------
        list[str]
            Allowed column names in result order.

        Raises
        ------
        ValueError
            If the view's table is missing from the manifest or exposes unknown columns in strict mode.
        """
        schema = inventory.get(view.table_key)
        if schema is None:
            msg = f"View table_key not present in schema manifest: {view.table_key}"
            raise ValueError(msg)

        schema_cols = [c.name for c in schema.columns]
        if not view.columns:
            return schema_cols

        unknown = sorted(set(view.columns) - set(schema_cols))
        mode = self.settings.schema_enforcement.lower()
        if unknown and mode == "strict":
            msg = f"Semantic view {view.id} exposes unknown columns: {unknown}"
            raise ValueError(msg)
        if unknown and mode == "warn":
            LOG.warning(
                "serving.semantic.columns.unknown view_id=%s table_key=%s unknown=%s",
                view.id,
                view.table_key,
                unknown,
            )
            return [c for c in view.columns if c in schema_cols]
        if mode == "off":
            return list(view.columns)

        return list(view.columns)

    def _execute_sql(
        self,
        *,
        con: DuckDBConnection,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> list[dict[str, object]]:
        engine = self.settings.result_engine.lower()
        backend = MinimalStorageGateway(con).policy
        result = backend.execute_sql(sql, params=params)

        if engine == "polars" and pl is not None:
            df_pl = result.pl()
            return _sanitize_rows(df_pl.to_dicts())

        if engine == "polars" and pl is None:
            LOG.warning("polars not installed; falling back to pandas result extraction")

        df_pd = result.df()
        sanitized = df_pd.astype("object").where(pd.notna(df_pd), None)
        return sanitized.to_dict(orient="records")

    def _execute_semantic_plan(
        self,
        *,
        con: DuckDBConnection,
        plan: SemanticQueryPlan,
    ) -> list[dict[str, object]]:
        ibis_con = ibis.duckdb.from_connection(con)
        expr = build_query(ibis_con=ibis_con, plan=plan)
        sql = ibis_con.compile(expr)
        return self._execute_sql(con=con, sql=sql)

    def catalog(self) -> dict[str, object]:
        """List all available semantic views.

        Returns
        -------
        dict[str, object]
            Catalog response with version, snapshot, and views.
        """
        registry = self._load_registry()
        pointer = self.db.current_pointer()

        return {
            "version": registry.version,
            "snapshot": {"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
            "views": [
                {
                    "id": v.id,
                    "table_key": v.table_key,
                    "entity": v.entity,
                    "grain": v.grain,
                    "description": v.description,
                    "column_count": len(v.columns),
                }
                for v in registry.views
                if not v.deprecated
            ],
        }

    def describe(self, view_id: str) -> dict[str, object]:
        """Describe a single semantic view.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        dict[str, object]
            View description with schema details.
        """
        registry = self._load_registry()
        inventory = self._load_inventory()
        pointer = self.db.current_pointer()

        view = registry.by_id(view_id)
        table_schema = inventory.get(view.table_key)

        column_types: dict[str, str] = {}
        if table_schema is not None:
            column_types = {c.name: c.type for c in table_schema.columns}

        return {
            "id": view.id,
            "table_key": view.table_key,
            "kind": view.kind,
            "entity": view.entity,
            "grain": view.grain,
            "description": view.description,
            "primary_key": view.primary_key,
            "columns": view.columns,
            "column_types": column_types,
            "joins": view.joins,
            "defaults": view.defaults.model_dump(mode="json"),
            "deprecated": view.deprecated,
            "replaced_by": view.replaced_by,
            "snapshot": {"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
        }

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
        """Execute a semantic view query.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.

        Returns
        -------
        SemanticQueryResponse
            Query results.
        """
        registry = self._load_registry()
        inventory = self._load_inventory()
        view = registry.by_id(request.view_id)

        allowed_columns = self._resolve_allowed_columns(view=view, inventory=inventory)
        columns = request.select if request.select else allowed_columns

        effective_limit = request.limit if request.limit else view.defaults.limit
        effective_order = request.order_by if request.order_by else view.defaults.order_by

        query_limit = effective_limit + 1
        plan = SemanticQueryPlan(
            table_key=view.table_key,
            columns=columns,
            allowed_columns=frozenset(allowed_columns),
            filters=request.filters,
            order_by=effective_order,
            limit=query_limit,
            offset=request.offset,
        )

        with self.db.connect() as (con, pointer):
            rows = self._execute_semantic_plan(con=con, plan=plan)

        truncated = len(rows) > effective_limit
        if truncated:
            rows = rows[:effective_limit]

        return SemanticQueryResponse(
            view_id=request.view_id,
            columns=columns,
            rows=rows,
            truncated=truncated,
            snapshot={"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
        )

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse:
        """Return compiled SQL and DuckDB EXPLAIN plan for a semantic query.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.

        Returns
        -------
        SemanticExplainResponse
            Explain output including compiled SQL and plan text.
        """
        registry = self._load_registry()
        inventory = self._load_inventory()
        view = registry.by_id(request.view_id)

        allowed_columns = self._resolve_allowed_columns(view=view, inventory=inventory)
        columns = request.select if request.select else allowed_columns

        effective_limit = request.limit if request.limit else view.defaults.limit
        effective_order = request.order_by if request.order_by else view.defaults.order_by

        plan = SemanticQueryPlan(
            table_key=view.table_key,
            columns=columns,
            allowed_columns=frozenset(allowed_columns),
            filters=request.filters,
            order_by=effective_order,
            limit=effective_limit,
            offset=request.offset,
        )

        with self.db.connect() as (con, pointer):
            ibis_con = ibis.duckdb.from_connection(con)
            compiled = ibis_con.compile(build_query(ibis_con=ibis_con, plan=plan))

            raw_rows = MinimalStorageGateway(con).policy.execute_sql(f"EXPLAIN {compiled}").fetchall()
            plan_text = _format_explain_rows(raw_rows)

        return SemanticExplainResponse(
            view_id=request.view_id,
            sql=compiled,
            plan=plan_text,
            snapshot={"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
        )

    def meta(self) -> dict[str, object]:
        """Return serving metadata for /meta endpoint and tools.

        Returns
        -------
        dict[str, object]
            Comprehensive serving metadata.
        """
        registry = self._load_registry()
        spec = self._load_buildspec()
        pointer = self.db.current_pointer()

        tables = sum(1 for d in spec.datasets if not d.table_key.startswith("docs.v_"))
        views = sum(1 for d in spec.datasets if d.table_key.startswith("docs.v_"))

        return {
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
            "published_at": pointer.published_at.isoformat(),
            "semantic_layer_version": pointer.semantic_layer_version,
            "buildspec_hash": spec.buildspec_hash,
            "buildspec_version": spec.spec_version,
            "duckdb": {"db_path": str(pointer.db_path), "read_only": True},
            "semantic_views": [
                {"id": v.id, "table_key": v.table_key, "entity": v.entity, "grain": v.grain}
                for v in registry.views
                if not v.deprecated
            ],
            "datasets": [
                {"table_key": dataset.table_key, "schema_hash": dataset.schema_hash}
                for dataset in spec.datasets
            ],
            "targets": [
                {
                    "name": t.name,
                    "domain": t.domain,
                    "impl_kind": t.impl_kind,
                    "deps": list(t.deps),
                    "outputs": list(t.outputs),
                    "artifacts": [
                        {"name": artifact.name, "kind": artifact.kind} for artifact in t.artifacts
                    ],
                }
                for t in spec.targets
            ],
            "schema_inventory": {"tables": tables, "views": views},
        }

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse:
        """Search code metadata using `docs.search_documents` (FTS when available).

        Parameters
        ----------
        request
            Search request parameters.

        Returns
        -------
        SearchQueryResponse
            Search results with stable ranking when the FTS index is available.
        """
        engine = self.settings.result_engine.lower()

        with self.db.connect() as (con, pointer):
            backend = MinimalStorageGateway(con).policy
            if not backend.table_exists(schema=_SEARCH_TABLE_SCHEMA, table=_SEARCH_TABLE_NAME):
                return SearchQueryResponse(
                    query=request.query,
                    results=[],
                    truncated=False,
                    snapshot={
                        "repo": pointer.repo,
                        "commit": pointer.commit,
                        "run_id": pointer.run_id,
                    },
                    engine=engine,
                )

            row = backend.execute_sql(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
                [_SEARCH_FTS_SCHEMA],
            ).fetchone()
            fts_available = row is not None

            query_limit = request.limit + 1
            if fts_available and request.kinds:
                sql = _SQL_SEARCH_FTS_KINDS
                params: list[object] = [request.query, request.kinds, query_limit, request.offset]
            elif fts_available:
                sql = _SQL_SEARCH_FTS
                params = [request.query, query_limit, request.offset]
            elif request.kinds:
                sql = _SQL_SEARCH_LIKE_KINDS
                params = [
                    request.query,
                    request.query,
                    request.query,
                    request.kinds,
                    query_limit,
                    request.offset,
                ]
            else:
                sql = _SQL_SEARCH_LIKE
                params = [request.query, request.query, request.query, query_limit, request.offset]

            rows = self._execute_sql(con=con, sql=sql, params=params)

        truncated = len(rows) > request.limit
        if truncated:
            rows = rows[: request.limit]

        results = [SearchResult.model_validate(row) for row in rows]
        return SearchQueryResponse(
            query=request.query,
            results=results,
            truncated=truncated,
            snapshot={"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
            engine=engine,
        )


__all__ = ["SemanticQueryKernel"]
