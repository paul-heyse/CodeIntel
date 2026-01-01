"""Legacy DuckDB relation IO helpers for Hamilton (deprecated)."""

from __future__ import annotations

from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.core.duckdb_types import DuckDBRelation
from codeintel.core.gateway import BuildGateway


def load_dataset_relation(*, gateway: BuildGateway, ref: DatasetRef) -> DuckDBRelation:
    """Reject relation-based dataset loaders.

    Parameters
    ----------
    gateway
        Storage gateway (unused).
    ref
        Dataset reference (unused).

    Raises
    ------
    RuntimeError
        Always raised because relation-based loaders are deprecated.
    """
    _ = (gateway, ref)
    msg = (
        "load_dataset_relation is deprecated for inference-first pipelines. "
        "Use dataset-backed loaders instead."
    )
    raise RuntimeError(msg)


__all__ = ["load_dataset_relation"]
