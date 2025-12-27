"""Loader node helpers for native Hamilton DAGs."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.io.duckdb_relation_adapter import load_dataset_relation
from codeintel.build.hamilton.naming import dataset_node, to_node_name
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.tagging import tag_loader_query
from codeintel.storage.gateway import DuckDBRelation


def _default_loader_name(*, target: str, table_key: str) -> str:
    return to_node_name(f"{target}.{table_key}", prefix="l")


def _loader_signature(*, dataset_param: str) -> inspect.Signature:
    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            dataset_param,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=DatasetRef,
        ),
    ]
    return inspect.Signature(params, return_annotation=DuckDBRelation)


def load_table(
    *,
    domain: str,
    target: str,
    table_key: str,
    node_name: str | None = None,
) -> Callable[..., DuckDBRelation]:
    """Build a tagged loader node for a dataset relation.

    Returns
    -------
    Callable[..., DuckDBRelation]
        Hamilton node that loads the dataset as a DuckDB relation.
    """
    resolved_node_name = node_name or _default_loader_name(target=target, table_key=table_key)
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> DuckDBRelation:
        dataset_ref = kwargs.get(dataset_param)
        if not isinstance(dataset_ref, DatasetRef):
            msg = f"Expected DatasetRef for {dataset_param}, got {type(dataset_ref)}"
            raise TypeError(msg)
        return load_dataset_relation(gateway=env.gateway, ref=dataset_ref)

    loader = set_signature(loader, _loader_signature(dataset_param=dataset_param))
    loader.__name__ = resolved_node_name
    loader.__doc__ = f"Load {table_key} as a DuckDB relation."
    return tag_loader_query(domain=domain, target=target, table_key=table_key)(loader)


def load_query(
    *,
    domain: str,
    target: str,
    table_key: str,
    sql: str,
    node_name: str | None = None,
) -> Callable[..., DuckDBRelation]:
    """Build a tagged loader node for a SQL query with dataset dependencies.

    Returns
    -------
    Callable[..., DuckDBRelation]
        Hamilton node that executes the SQL using DuckDB.

    Raises
    ------
    ValueError
        If the provided SQL string is empty.
    """
    if not isinstance(sql, str) or not sql:
        msg = "load_query requires a non-empty SQL string"
        raise ValueError(msg)
    resolved_node_name = node_name or _default_loader_name(target=target, table_key=table_key)
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> DuckDBRelation:
        dataset_ref = kwargs.get(dataset_param)
        if not isinstance(dataset_ref, DatasetRef):
            msg = f"Expected DatasetRef for {dataset_param}, got {type(dataset_ref)}"
            raise TypeError(msg)
        if dataset_ref.table_key != table_key:
            msg = (
                f"DatasetRef table_key mismatch for {resolved_node_name}: "
                f"{dataset_ref.table_key} != {table_key}"
            )
            raise ValueError(msg)
        return env.gateway.con.sql(sql)

    loader = set_signature(loader, _loader_signature(dataset_param=dataset_param))
    loader.__name__ = resolved_node_name
    loader.__doc__ = f"Load query for {table_key} as a DuckDB relation."
    return tag_loader_query(domain=domain, target=target, table_key=table_key)(loader)


__all__ = ["load_query", "load_table"]
