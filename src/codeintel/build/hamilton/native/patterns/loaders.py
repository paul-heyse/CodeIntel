"""Loader node helpers for native Hamilton DAGs."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import ibis.expr.types as ir

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.io.ibis_adapter import load_dataset_ibis
from codeintel.build.hamilton.naming import dataset_node, to_node_name
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.tagging import tag_loader_query


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
    return inspect.Signature(params, return_annotation=ir.Table)


def load_table(
    *,
    domain: str,
    target: str,
    table_key: str,
    node_name: str | None = None,
) -> Callable[..., ir.Table]:
    """Build a tagged loader node for a dataset table.

    Returns
    -------
    Callable[..., ir.Table]
        Hamilton node that loads the dataset as an Ibis expression.
    """
    resolved_node_name = node_name or _default_loader_name(target=target, table_key=table_key)
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> ir.Table:
        dataset_ref = kwargs.get(dataset_param)
        if not isinstance(dataset_ref, DatasetRef):
            msg = f"Expected DatasetRef for {dataset_param}, got {type(dataset_ref)}"
            raise TypeError(msg)
        return load_dataset_ibis(gateway=env.gateway, ref=dataset_ref)

    loader = set_signature(loader, _loader_signature(dataset_param=dataset_param))
    loader.__name__ = resolved_node_name
    loader.__doc__ = f"Load {table_key} as an Ibis table expression."
    return tag_loader_query(domain=domain, target=target, table_key=table_key)(loader)


def load_query(
    *,
    domain: str,
    target: str,
    table_key: str,
    sql: str,
    node_name: str | None = None,
) -> Callable[..., ir.Table]:
    """Build a tagged loader node for a SQL query with dataset dependencies.

    Returns
    -------
    Callable[..., ir.Table]
        Hamilton node that executes the SQL using the DuckDB connection.
    """
    if not isinstance(sql, str) or not sql:
        msg = "load_query requires a non-empty SQL string"
        raise ValueError(msg)
    resolved_node_name = node_name or _default_loader_name(target=target, table_key=table_key)
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> ir.Table:
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
        return env.gateway.ibis.con.sql(sql)

    loader = set_signature(loader, _loader_signature(dataset_param=dataset_param))
    loader.__name__ = resolved_node_name
    loader.__doc__ = f"Load query for {table_key} as an Ibis table expression."
    return tag_loader_query(domain=domain, target=target, table_key=table_key)(loader)


__all__ = ["load_query", "load_table"]
