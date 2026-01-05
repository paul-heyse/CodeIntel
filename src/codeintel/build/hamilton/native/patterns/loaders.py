"""Loader node helpers for native Hamilton DAGs."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.naming import dataset_node, to_node_name
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.tagging import tag_loader_query
from codeintel.build.schemas import get_contract_for_table_key
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe, reader_to_table
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.core.columnar.schema_alignment import align_table_to_contract
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import (
    ArrowDatasetScanOptions,
    scan_dataset_reader,
)
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.validation.mode import ContractValidationMode
from codeintel.core.validation.profiles import ValidationProfile
from codeintel.core.validation.schema_constraints import schema_errors, schema_metadata_errors
from codeintel.storage.validation.columnar import (
    ColumnarValidationContext,
    ValidationMode,
    validate_table,
)

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema

log = logging.getLogger(__name__)


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
    return inspect.Signature(params, return_annotation=pa.Table)


def _snapshot_dir(
    *,
    env: BuildEnv,
    table_key: str,
    snapshot_id: str,
) -> Path:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        msg = "BuildEnv.paths.dataset_root_dir is required for dataset loaders"
        raise ValueError(msg)
    return dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )


def _scan_options(env: BuildEnv, table_key: str) -> ArrowDatasetScanOptions:
    schema_service = get_schema_service()
    arrow_schema = schema_service.get_arrow_schema(table_key)
    if arrow_schema is None:
        table_schema = schema_service.get_table_schema(table_key)
        if table_schema is not None:
            arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    return ArrowDatasetScanOptions(
        batch_size=DEFAULT_ARROW_BATCH_SIZE,
        schema=arrow_schema,
        schema_promote_options=env.settings.schema_promote_options,
    )


def _contract_schema_for_table(
    table_key: str,
) -> tuple[TableSchema, pa.Schema]:
    schema_service = get_schema_service()
    table_schema = schema_service.get_table_schema(table_key)
    if table_schema is None:
        msg = f"Missing TableSchema for {table_key}"
        raise ValueError(msg)
    arrow_schema = schema_service.get_arrow_schema(table_key)
    if arrow_schema is None:
        arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    return table_schema, arrow_schema


def _validation_profile_for_table(table_key: str) -> ValidationProfile | None:
    try:
        contract = get_contract_for_table_key(table_key)
    except (KeyError, ValueError, RuntimeError):
        return None
    return contract.validation_profile


def _validation_mode(env: BuildEnv) -> ValidationMode:
    if env.validation_mode == ContractValidationMode.OFF:
        return "skip"
    if env.validation_mode == ContractValidationMode.LENIENT:
        return "warn"
    return "strict"


def _handle_schema_errors(
    *,
    table_key: str,
    errors: list[str],
    mode: ValidationMode,
) -> None:
    if not errors or mode == "skip":
        return
    if mode == "warn":
        for error in errors:
            log.warning("Input schema mismatch for %s: %s", table_key, error)
        return
    message = f"Schema mismatch for {table_key}: " + "; ".join(errors)
    raise ValueError(message)


def load_snapshot_tabular(
    *,
    env: BuildEnv,
    table_key: str,
    snapshot_id: str,
) -> pa.Table:
    """Load a dataset snapshot as an Arrow table.

    Parameters
    ----------
    env
        Build environment with dataset root paths.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.

    Returns
    -------
    pyarrow.Table
        Materialized table for the snapshot dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset snapshot cannot be located on disk.
    """
    snapshot_dir = _snapshot_dir(env=env, table_key=table_key, snapshot_id=snapshot_id)
    try:
        reader = scan_dataset_reader(
            dataset_root=env.paths.dataset_root_dir,
            table_key=table_key,
            snapshot_id=snapshot_id,
            options=_scan_options(env, table_key),
        )
    except FileNotFoundError as exc:
        msg = f"Dataset snapshot not found: {snapshot_dir}"
        raise FileNotFoundError(msg) from exc
    except (OSError, ValueError, TypeError, pa.ArrowInvalid) as exc:
        msg = f"Dataset snapshot not found: {snapshot_dir}"
        raise FileNotFoundError(msg) from exc
    table_schema, contract_schema = _contract_schema_for_table(table_key)
    validation_mode = _validation_mode(env)
    table = reader_to_table(reader)
    if validation_mode != "skip":
        schema_for_errors = table.schema
        if contract_schema.metadata is not None:
            schema_for_errors = table.schema.with_metadata(contract_schema.metadata)
        errors = schema_errors(table_schema, schema_for_errors)
        errors.extend(schema_metadata_errors(table.schema))
        _handle_schema_errors(
            table_key=table_key,
            errors=errors,
            mode=validation_mode,
        )
    try:
        aligned = align_table_to_contract(
            table,
            contract_schema,
            schema_promote_options=env.settings.schema_promote_options,
        )
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        _handle_schema_errors(
            table_key=table_key,
            errors=[str(exc)],
            mode=validation_mode,
        )
        return table
    if validation_mode == "skip":
        return aligned
    context = ColumnarValidationContext(
        table_schema=table_schema,
        schema_observation=None,
        validation_profile=_validation_profile_for_table(table_key),
    )
    return validate_table(
        table_key,
        aligned,
        context=context,
        mode=validation_mode,
    )


def load_snapshot_lazyframe(
    *,
    env: BuildEnv,
    table_key: str,
    snapshot_id: str,
) -> TabularFrame:
    """Load a dataset snapshot as a Polars LazyFrame.

    Parameters
    ----------
    env
        Build environment with dataset root paths.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.

    Returns
    -------
    polars.LazyFrame
        Lazy frame backed by the snapshot dataset.

    """
    reader = load_snapshot_tabular(
        env=env,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return arrow_reader_to_lazyframe(reader)


def load_table(
    *,
    domain: str,
    target: str,
    table_key: str,
    node_name: str | None = None,
) -> Callable[..., InferableTabularInput]:
    """Build a tagged loader node for a dataset relation.

    Returns
    -------
    Callable[..., InferableTabularInput]
        Hamilton node that loads the dataset as a tabular input.

    Notes
    -----
    DatasetRef.commit overrides env.commit for seeded snapshot loads.
    """
    resolved_node_name = node_name or _default_loader_name(target=target, table_key=table_key)
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> InferableTabularInput:
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
        snapshot_id = dataset_ref.commit or env.commit
        if not snapshot_id:
            msg = f"Missing snapshot_id for {table_key}"
            raise ValueError(msg)
        return load_snapshot_tabular(
            env=env,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )

    loader = set_signature(loader, _loader_signature(dataset_param=dataset_param))
    loader.__name__ = resolved_node_name
    loader.__doc__ = f"Load {table_key} as a dataset-backed Arrow table."
    return tag_loader_query(domain=domain, target=target, table_key=table_key)(loader)


__all__ = ["load_snapshot_lazyframe", "load_snapshot_tabular", "load_table"]
