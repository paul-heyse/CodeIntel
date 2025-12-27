"""DuckDB relation IO helpers for Hamilton."""

from __future__ import annotations

from duckdb import ColumnExpression, ConstantExpression

from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.storage.duckdb_types import DuckDBRelation
from codeintel.storage.gateway import StorageGateway


def _relation_has_repo_commit_columns(relation: DuckDBRelation) -> bool:
    columns = getattr(relation, "columns", ())
    return "repo" in columns and "commit" in columns


def load_dataset_relation(*, gateway: StorageGateway, ref: DatasetRef) -> DuckDBRelation:
    """Load a dataset as a DuckDB relation scoped to repo/commit if available.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    ref
        Dataset reference with table key and snapshot identity.

    Returns
    -------
    DuckDBRelation
        Relation for the dataset, optionally filtered by repo/commit.
    """
    relation = gateway.relation_from_table_key(ref.table_key)
    if ref.repo and ref.commit and _relation_has_repo_commit_columns(relation):
        relation = relation.filter(
            (ColumnExpression("repo") == ConstantExpression(ref.repo))
            & (ColumnExpression("commit") == ConstantExpression(ref.commit))
        )
    return relation


__all__ = ["load_dataset_relation"]
