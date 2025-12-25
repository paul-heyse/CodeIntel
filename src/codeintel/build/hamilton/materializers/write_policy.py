"""Helpers for applying table-level write policies to materialization options."""

from __future__ import annotations

from collections.abc import Sequence

from sqlglot import exp

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materialize_options import (
    MaterializeOptionsConfig,
    materialize_options,
)
from codeintel.build.schemas.service import get_schema_service
from codeintel.core.schemas.primitives import TableSchema, TableWritePolicy
from codeintel.storage.warehouse import MaterializeOptions, UpsertConfig


def resolve_materialize_options(
    *,
    env: BuildEnv,
    target_name: str,
    table_key: str,
    input_hash: str | None,
    column_names: Sequence[str] | None = None,
) -> MaterializeOptions:
    """Resolve materialization options based on table write policy.

    Parameters
    ----------
    env
        Build environment providing snapshot.
    target_name
        Target name that owns the materialization.
    table_key
        Fully qualified table key to resolve schema policy for.
    input_hash
        Optional manifest input hash for observability.
    column_names
        Optional column names for validation against the schema.

    Returns
    -------
    MaterializeOptions
        Resolved options with write policy and upsert configuration applied.
    """
    schema = get_schema_service().get_table_schema(table_key)
    policy = (
        schema.write_policy if schema and schema.write_policy is not None else TableWritePolicy()
    )
    resolved_columns = _resolve_columns(schema, column_names)
    upsert_config = _build_upsert_config(policy, schema, resolved_columns)
    return materialize_options(
        env,
        owner_target=target_name,
        config=MaterializeOptionsConfig(
            mode=policy.mode,
            replace_scope=policy.replace_scope,
            input_hash=input_hash,
            upsert=upsert_config,
            use_staging=policy.use_staging,
        ),
    )


def _resolve_columns(
    schema: TableSchema | None,
    column_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if column_names is not None:
        return tuple(column_names)
    if schema is None:
        return ()
    return tuple(schema.column_names())


def _build_upsert_config(
    policy: TableWritePolicy,
    schema: TableSchema | None,
    resolved_columns: tuple[str, ...],
) -> UpsertConfig | None:
    if policy.mode != "upsert":
        return None

    conflict_columns = policy.conflict_columns
    if conflict_columns is None:
        conflict_columns = tuple(schema.primary_key) if schema is not None else ()

    if not conflict_columns:
        msg = "Upsert write policy requires conflict_columns or primary key"
        raise ValueError(msg)

    update_columns = policy.update_columns
    update_condition = None
    if policy.hash_column is not None:
        update_condition = _hash_update_condition(policy.hash_column)
        if update_columns is not None and policy.hash_column not in update_columns:
            update_columns = (*update_columns, policy.hash_column)

    _validate_upsert_columns(conflict_columns, update_columns, resolved_columns)

    return UpsertConfig(
        conflict_columns=conflict_columns,
        update_columns=update_columns,
        update_condition=update_condition,
    )


def _validate_upsert_columns(
    conflict_columns: tuple[str, ...],
    update_columns: tuple[str, ...] | None,
    resolved_columns: tuple[str, ...],
) -> None:
    if not resolved_columns:
        return

    column_set = set(resolved_columns)
    missing_conflicts = [col for col in conflict_columns if col not in column_set]
    if missing_conflicts:
        msg = f"Upsert conflict columns not in schema: {missing_conflicts}"
        raise ValueError(msg)

    if update_columns is None:
        return

    missing_updates = [col for col in update_columns if col not in column_set]
    if missing_updates:
        msg = f"Upsert update columns not in schema: {missing_updates}"
        raise ValueError(msg)


def _hash_update_condition(hash_column: str) -> exp.Expression:
    empty_literal = exp.Literal.string("")
    existing = exp.Coalesce(
        this=exp.Column(this=exp.to_identifier(hash_column)),
        expressions=[empty_literal],
    )
    incoming = exp.Coalesce(
        this=exp.Column(
            this=exp.to_identifier(hash_column),
            table=exp.to_identifier("excluded"),
        ),
        expressions=[empty_literal],
    )
    return exp.Where(this=exp.NEQ(this=existing, expression=incoming))


__all__ = ["resolve_materialize_options"]
