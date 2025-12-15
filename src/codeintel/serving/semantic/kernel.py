"""Semantic query kernel - unified API for HTTP and MCP.

The kernel provides a single entry point for semantic layer queries, used by
both FastAPI routes and MCP tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ibis
import pandas as pd

from codeintel.build.spec import buildspec_from_json
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.models import SemanticQueryResponse
from codeintel.serving.semantic.query_builder import SemanticQueryPlan, build_query
from codeintel.serving.semantic.registry import SemanticRegistry

if TYPE_CHECKING:
    from codeintel.build.spec import BuildSpec
    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.semantic.models import SemanticQueryRequest


@dataclass
class SemanticQueryKernel:
    """Unified query kernel for semantic layer access.

    Parameters
    ----------
    db
        Database manager for connection access.
    """

    db: ServingDBManager

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
        view = registry.by_id(request.view_id)

        columns = request.select if request.select else view.columns
        allowed = set(view.columns)

        effective_limit = request.limit if request.limit else view.defaults.limit
        effective_order = request.order_by if request.order_by else view.defaults.order_by

        query_limit = effective_limit + 1
        plan = SemanticQueryPlan(
            table_key=view.table_key,
            columns=columns,
            allowed_columns=frozenset(allowed),
            filters=request.filters,
            order_by=effective_order,
            limit=query_limit,
            offset=request.offset,
        )

        with self.db.connect() as (con, pointer):
            ibis_con = ibis.duckdb.from_connection(con)
            expr = build_query(ibis_con=ibis_con, plan=plan)
            df = pd.DataFrame(expr.execute())
            sanitized = df.astype("object").where(pd.notna(df), None)
            rows = sanitized.to_dict(orient="records")

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


__all__ = ["SemanticQueryKernel"]
