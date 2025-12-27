"""Centralized query planning for semantic operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.schemas.hashing import schema_hash
from codeintel.serving.errors import SemanticColumnNotFoundError
from codeintel.serving.semantic.query_builder import SemanticQueryPlan, build_query
from codeintel.storage.gateway import ibis_facade
from codeintel.storage.queries.safe import (
    SqlIngressPolicy,
    UnsafeSqlError,
    assert_select_perimeter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.db.manager import ServingDBManager, ServingSnapshotContext
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.semantic.inventory import SchemaInventory
    from codeintel.serving.semantic.models import (
        FilterSpec,
        SemanticExportRequest,
        SemanticQueryRequest,
        SemanticViewSpec,
    )
    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.warehouse import Warehouse

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedViewContext:
    """Resolved semantic view metadata for planning."""

    pointer: ServingSnapshotPointer
    view: SemanticViewSpec
    inventory: SchemaInventory
    allowed_columns: list[str]
    column_types: dict[str, ColumnType] | None


@dataclass(frozen=True, slots=True)
class PlanInputs:
    """Normalized query inputs for plan construction."""

    columns: list[str]
    filters: list[FilterSpec]
    order_by: list[str]
    offset: int


@dataclass(frozen=True, slots=True)
class SemanticQueryPlanner:
    """Resolve semantic views and build safe query plans.

    Parameters
    ----------
    db
        Serving DB manager for snapshot context access.
    settings
        Serving settings for schema enforcement and defaults.
    """

    db: ServingDBManager
    settings: ServingSettings

    def snapshot_context(self, pointer: ServingSnapshotPointer) -> ServingSnapshotContext:
        """Return the cached snapshot context for a pointer.

        Parameters
        ----------
        pointer
            Snapshot pointer describing the active artifact paths.

        Returns
        -------
        ServingSnapshotContext
            Cached registry/inventory/buildspec for the snapshot.
        """
        return self.db.snapshot_context(pointer)

    def resolve_view_context(
        self, *, pointer: ServingSnapshotPointer, view_id: str
    ) -> ResolvedViewContext:
        """Resolve view context for planning.

        Parameters
        ----------
        pointer
            Snapshot pointer.
        view_id
            Semantic view identifier.

        Returns
        -------
        ResolvedViewContext
            Resolved view metadata and allowed columns.
        """
        context = self.snapshot_context(pointer)
        inventory = context.inventory
        view = context.registry.by_id(view_id)
        allowed_columns = self._resolve_allowed_columns(view=view, inventory=inventory)
        column_types = _column_types_for_view(view=view, inventory=inventory)
        return ResolvedViewContext(
            pointer=pointer,
            view=view,
            inventory=inventory,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )

    @staticmethod
    def plan_inputs_for_query(
        *, ctx: ResolvedViewContext, request: SemanticQueryRequest
    ) -> tuple[PlanInputs, int]:
        """Return plan inputs and effective limit for a query request.

        Returns
        -------
        tuple[PlanInputs, int]
            Normalized plan inputs and the effective limit to apply.

        Raises
        ------
        SemanticColumnNotFoundError
            If requested columns are not allowed for the view.
        """
        if request.select:
            unknown = sorted(set(request.select) - set(ctx.allowed_columns))
            if unknown:
                raise SemanticColumnNotFoundError(ctx.view.id, unknown[0])
            columns = list(request.select)
        else:
            columns = ctx.allowed_columns
        effective_limit = request.limit if request.limit else ctx.view.defaults.limit
        effective_order = request.order_by if request.order_by else ctx.view.defaults.order_by
        inputs = PlanInputs(
            columns=columns,
            filters=request.filters,
            order_by=effective_order,
            offset=request.offset,
        )
        return inputs, effective_limit

    @staticmethod
    def plan_inputs_for_export(
        *, ctx: ResolvedViewContext, request: SemanticExportRequest
    ) -> tuple[PlanInputs, int]:
        """Return plan inputs and effective limit for an export request.

        Returns
        -------
        tuple[PlanInputs, int]
            Normalized plan inputs and the effective limit to apply.

        Raises
        ------
        SemanticColumnNotFoundError
            If requested columns are not allowed for the view.
        """
        if request.select:
            unknown = sorted(set(request.select) - set(ctx.allowed_columns))
            if unknown:
                raise SemanticColumnNotFoundError(ctx.view.id, unknown[0])
            columns = list(request.select)
        else:
            columns = ctx.allowed_columns

        inputs = PlanInputs(
            columns=columns,
            filters=request.filters,
            order_by=request.order_by,
            offset=request.offset,
        )
        return inputs, request.limit

    @staticmethod
    def build_plan(
        *, ctx: ResolvedViewContext, inputs: PlanInputs, limit: int
    ) -> SemanticQueryPlan:
        """Build a semantic query plan from resolved context and inputs.

        Returns
        -------
        SemanticQueryPlan
            Planned query parameters ready for compilation.
        """
        return SemanticQueryPlan(
            table_key=ctx.view.table_key,
            columns=inputs.columns,
            allowed_columns=frozenset(ctx.allowed_columns),
            filters=inputs.filters,
            order_by=inputs.order_by,
            limit=limit,
            offset=inputs.offset,
        )

    @staticmethod
    def schema_hash_for_table_key(*, inventory: SchemaInventory, table_key: str) -> str | None:
        """Return a schema hash for a table key, if available.

        Returns
        -------
        str | None
            Schema hash when the table key is known, otherwise None.
        """
        schema = inventory.get(table_key)
        if schema is None:
            return None
        return schema_hash(schema)

    @staticmethod
    def compile_plan(
        *,
        warehouse: Warehouse,
        plan: SemanticQueryPlan,
        column_types: dict[str, ColumnType] | None,
        cleanup_temp_tables: bool,
    ) -> tuple[str, tuple[str, ...]]:
        """Compile a semantic query plan into safe SQL.

        Returns
        -------
        tuple[str, tuple[str, ...]]
            Compiled SQL string and created temp table names.

        Raises
        ------
        ValueError
            If the compiled SQL is unsafe.
        """
        ibis_con = ibis_facade.backend(warehouse.gateway)
        built = build_query(ibis_con=ibis_con, plan=plan, column_types=column_types)
        try:
            compiled = built.compile_sql(ibis_con)
            assert_select_perimeter(compiled, policy=SqlIngressPolicy())
        except UnsafeSqlError as exc:
            cleanup_temp_tables_if_needed(
                con=warehouse.gateway.con,
                temp_tables=built.temp_tables,
            )
            raise ValueError(str(exc)) from exc
        except Exception:
            cleanup_temp_tables_if_needed(
                con=warehouse.gateway.con,
                temp_tables=built.temp_tables,
            )
            raise

        if cleanup_temp_tables:
            cleanup_temp_tables_if_needed(
                con=warehouse.gateway.con,
                temp_tables=built.temp_tables,
            )

        return compiled, built.temp_tables

    def compile_export(
        self,
        *,
        warehouse: Warehouse,
        pointer: ServingSnapshotPointer,
        request: SemanticExportRequest,
    ) -> tuple[str, list[str], tuple[str, ...]]:
        """Compile an export request into SQL plus column order and temp tables.

        Returns
        -------
        tuple[str, list[str], tuple[str, ...]]
            SQL string, column order, and temp table names.
        """
        resolved = self.resolve_view_context(pointer=pointer, view_id=request.view_id)
        inputs, effective_limit = self.plan_inputs_for_export(ctx=resolved, request=request)
        plan = self.build_plan(ctx=resolved, inputs=inputs, limit=effective_limit)
        sql, temp_tables = self.compile_plan(
            warehouse=warehouse,
            plan=plan,
            column_types=resolved.column_types,
            cleanup_temp_tables=False,
        )
        return sql, inputs.columns, temp_tables

    def _resolve_allowed_columns(
        self,
        *,
        view: SemanticViewSpec,
        inventory: SchemaInventory,
    ) -> list[str]:
        """Resolve view columns after applying schema enforcement rules.

        Returns
        -------
        list[str]
            Allowed columns for the view.

        Raises
        ------
        ValueError
            If the view references unknown columns in strict mode or the schema is missing.
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


def _column_types_for_view(
    *,
    view: SemanticViewSpec,
    inventory: SchemaInventory,
) -> dict[str, ColumnType] | None:
    table_schema = inventory.get(view.table_key)
    if table_schema is None:
        return None
    return {col.name: col.type for col in table_schema.columns}


def cleanup_temp_tables_if_needed(*, con: object, temp_tables: Sequence[str]) -> None:
    """Unregister Ibis temp tables when supported by the backend."""
    unregister = getattr(con, "unregister", None)
    if not callable(unregister):
        return
    for table_name in temp_tables:
        unregister(table_name)


__all__ = [
    "PlanInputs",
    "ResolvedViewContext",
    "SemanticQueryPlanner",
    "cleanup_temp_tables_if_needed",
]
