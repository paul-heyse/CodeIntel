"""Loader node helpers for native Hamilton DAGs."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.naming import dataset_node, to_node_name
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.tagging import tag_loader_query
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.core.columnar.dataset_scanner import scan_dataset_lazyframe
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.datasets.paths import dataset_snapshot_dir


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
    return inspect.Signature(params, return_annotation=InferableTabularInput)


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

    Raises
    ------
    FileNotFoundError
        If the dataset snapshot cannot be located on disk.
    """
    snapshot_dir = _snapshot_dir(env=env, table_key=table_key, snapshot_id=snapshot_id)
    frame = scan_dataset_lazyframe(snapshot_dir, batch_size=DEFAULT_ARROW_BATCH_SIZE)
    if frame is None:
        msg = f"Dataset snapshot not found: {snapshot_dir}"
        raise FileNotFoundError(msg)
    return frame


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
        return load_snapshot_lazyframe(
            env=env,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )

    loader = set_signature(loader, _loader_signature(dataset_param=dataset_param))
    loader.__name__ = resolved_node_name
    loader.__doc__ = f"Load {table_key} as a dataset-backed LazyFrame."
    return tag_loader_query(domain=domain, target=target, table_key=table_key)(loader)


def load_query(
    *,
    domain: str,
    target: str,
    table_key: str,
    sql: str,
    node_name: str | None = None,
) -> Callable[..., InferableTabularInput]:
    """Build a tagged loader node for a SQL query with dataset dependencies.

    Raises
    ------
    RuntimeError
        Always raised because SQL-based loaders are deprecated.
    """
    _ = (domain, target, table_key, node_name, sql)
    msg = (
        "load_query is deprecated for inference-first pipelines. "
        "Use dataset-backed loaders or Hamilton view outputs instead."
    )
    raise RuntimeError(msg)


__all__ = ["load_query", "load_snapshot_lazyframe", "load_table"]
