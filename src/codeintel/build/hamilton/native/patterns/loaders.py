"""Loader node helpers for native Hamilton DAGs."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pyiceberg.expressions import (
    AlwaysTrue,
    And,
    BooleanExpression,
    EqualTo,
    Reference,
)
from pyiceberg.expressions.literals import literal

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.io.duckdb_relation_adapter import load_dataset_relation
from codeintel.build.hamilton.naming import dataset_node, to_node_name
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.tagging import tag_loader_query
from codeintel.build.tabular.types import TabularInput
from codeintel.core.config.view import SettingsView
from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.iceberg.guardrails import iceberg_enforced_table, require_iceberg_read
from codeintel.serving.semantic.iceberg_scans import (
    IcebergRefScanRequest,
    iceberg_scan_for_ref,
    iceberg_table_exists,
    resolve_iceberg_ref_for_identity,
)

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyiceberg.schema import Schema


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
    return inspect.Signature(params, return_annotation=TabularInput)


def _iceberg_has_column(table_schema: Schema, name: str) -> bool:
    schema = table_schema
    try:
        return schema.find_field(name) is not None
    except (KeyError, ValueError, AttributeError):
        return False


def _iceberg_row_filter_for_ref(
    *,
    table_schema: Schema,
    ref: DatasetRef,
) -> BooleanExpression:
    predicates: list[BooleanExpression] = []
    if ref.repo and _iceberg_has_column(table_schema, "repo"):
        predicates.append(EqualTo(Reference("repo"), literal(ref.repo)))
    if ref.commit and _iceberg_has_column(table_schema, "commit"):
        predicates.append(EqualTo(Reference("commit"), literal(ref.commit)))
    if not predicates:
        return AlwaysTrue()
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = And(combined, predicate)
    return combined


def _load_iceberg_scan(
    *,
    settings_view: SettingsView,
    table_key: str,
    ref: DatasetRef,
) -> object | None:
    iceberg = settings_view.build.iceberg
    if not iceberg.read_enabled:
        return None
    if not iceberg_table_exists(settings=iceberg, table_key=table_key):
        return None
    provider = IcebergCatalogProvider(iceberg)
    table = provider.load_table(table_key)
    row_filter = _iceberg_row_filter_for_ref(table_schema=table.schema(), ref=ref)
    run_id = ref.metadata.get("run_id")
    resolved_run_id = run_id if isinstance(run_id, str) and run_id else None
    ref_name = resolve_iceberg_ref_for_identity(
        run_id=resolved_run_id,
        commit=ref.commit,
        settings=iceberg,
    )
    scan_result = iceberg_scan_for_ref(
        request=IcebergRefScanRequest(
            table_key=table_key,
            selected_fields=(),
            row_filter=row_filter,
            ref_name=ref_name,
            settings=iceberg,
            batch_size=None,
            table=table,
        )
    )
    return scan_result.scan


def load_table(
    *,
    domain: str,
    target: str,
    table_key: str,
    node_name: str | None = None,
) -> Callable[..., TabularInput]:
    """Build a tagged loader node for a dataset relation.

    Returns
    -------
    Callable[..., TabularInput]
        Hamilton node that loads the dataset as a tabular input.
    """
    resolved_node_name = node_name or _default_loader_name(target=target, table_key=table_key)
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> TabularInput:
        dataset_ref = kwargs.get(dataset_param)
        if not isinstance(dataset_ref, DatasetRef):
            msg = f"Expected DatasetRef for {dataset_param}, got {type(dataset_ref)}"
            raise TypeError(msg)
        settings_view = SettingsView.from_build_env(env)
        if iceberg_enforced_table(
            settings=settings_view.build.iceberg,
            table_key=table_key,
        ):
            require_iceberg_read(settings=settings_view.build.iceberg, table_key=table_key)
            scan = _load_iceberg_scan(
                settings_view=settings_view,
                table_key=table_key,
                ref=dataset_ref,
            )
            if scan is None:
                msg = f"Iceberg table missing for enforced table: {table_key}"
                raise ValueError(msg)
            return scan
        scan = _load_iceberg_scan(
            settings_view=settings_view,
            table_key=table_key,
            ref=dataset_ref,
        )
        if scan is not None:
            return scan
        return load_dataset_relation(gateway=env.gateway, ref=dataset_ref)

    loader = set_signature(loader, _loader_signature(dataset_param=dataset_param))
    loader.__name__ = resolved_node_name
    loader.__doc__ = f"Load {table_key} as a tabular input."
    return tag_loader_query(domain=domain, target=target, table_key=table_key)(loader)


def load_query(
    *,
    domain: str,
    target: str,
    table_key: str,
    sql: str,
    node_name: str | None = None,
) -> Callable[..., TabularInput]:
    """Build a tagged loader node for a SQL query with dataset dependencies.

    Returns
    -------
    Callable[..., TabularInput]
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

    def loader(env: BuildEnv, **kwargs: object) -> TabularInput:
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
