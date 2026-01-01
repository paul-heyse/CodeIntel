"""Pure compute functions for data model extraction.

This module provides pure compute functions that return row data without
performing any database writes. The materialization is handled by the
Hamilton native module in `build/hamilton/native/analytics/data_models.py`.

The functions extract structured data models from class definitions,
returning structured result containers that can be materialized to DuckDB
tables by the build system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.analytics.data_models.core import (
    DATA_MODEL_FIELDS_COLS,
    DATA_MODEL_RELATIONSHIPS_COLS,
    DATA_MODELS_COLS,
    _attach_relationships,
    _doc_map,
    _gather_models_for_path,
    _load_class_metadata,
)
from codeintel.core.paths import normalize_path
from codeintel.storage.repositories import RepositoryFactory

if TYPE_CHECKING:
    from codeintel.build.analytics.data_models.core import ClassMeta, ModelRecord
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataModelsResult:
    """Result container for data models computation.

    Contains row data for all three data model tables without performing writes.
    The rows are tuples matching the column specifications in the schema.

    Attributes
    ----------
    model_rows
        Rows for analytics.data_models table.
    field_rows
        Rows for analytics.data_model_fields table.
    relationship_rows
        Rows for analytics.data_model_relationships table.
    """

    model_rows: tuple[tuple[object, ...], ...]
    field_rows: tuple[tuple[object, ...], ...]
    relationship_rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class DataModelsInputs:
    """Input payload for data model computation."""

    class_metas: tuple[ClassMeta, ...]
    doc_map: dict[tuple[str, str], tuple[str | None, str | None]]


def _build_model_rows(
    models: list[ModelRecord],
    snapshot: SnapshotRef,
    now: datetime,
) -> list[tuple[object, ...]]:
    """Build row tuples for analytics.data_models table.

    Parameters
    ----------
    models
        List of extracted model records.
    snapshot
        Repository and commit snapshot reference.
    now
        Timestamp for created_at field.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples matching DATA_MODELS_COLS.
    """
    return [
        (
            snapshot.repo,
            snapshot.commit,
            model.model_id,
            model.goid,
            model.model_name,
            model.module,
            model.rel_path,
            model.model_kind,
            json.dumps(model.base_classes),
            model.doc_short,
            model.doc_long,
            now,
        )
        for model in models
    ]


def _build_field_rows(
    models: list[ModelRecord],
    snapshot: SnapshotRef,
    now: datetime,
) -> list[tuple[object, ...]]:
    """Build row tuples for analytics.data_model_fields table.

    Parameters
    ----------
    models
        List of extracted model records.
    snapshot
        Repository and commit snapshot reference.
    now
        Timestamp for created_at field.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples matching DATA_MODEL_FIELDS_COLS.
    """
    return [
        (
            snapshot.repo,
            snapshot.commit,
            model.model_id,
            field_spec.name,
            field_spec.type,
            field_spec.required,
            field_spec.has_default,
            field_spec.default_expr,
            json.dumps(field_spec.constraints),
            field_spec.source,
            model.rel_path,
            field_spec.lineno,
            now,
        )
        for model in models
        for field_spec in model.fields
    ]


def _build_relationship_rows(
    models: list[ModelRecord],
    snapshot: SnapshotRef,
    now: datetime,
) -> list[tuple[object, ...]]:
    """Build row tuples for analytics.data_model_relationships table.

    Parameters
    ----------
    models
        List of extracted model records.
    snapshot
        Repository and commit snapshot reference.
    now
        Timestamp for created_at field.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples matching DATA_MODEL_RELATIONSHIPS_COLS.
    """
    return [
        (
            snapshot.repo,
            snapshot.commit,
            model.model_id,
            rel.target_model_id,
            rel.target_module,
            rel.target_model_name,
            rel.field_name,
            rel.kind,
            rel.multiplicity,
            rel.via,
            json.dumps(rel.evidence) if rel.evidence else None,
            rel.rel_path,
            rel.lineno,
            now,
        )
        for model in models
        for rel in model.relationships
    ]


def load_data_models_inputs(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> DataModelsInputs:
    """Load storage-backed inputs required for data model computation.

    Returns
    -------
    DataModelsInputs
        Loaded class metadata and docstring mappings.
    """
    repo = RepositoryFactory(gateway, repo=snapshot.repo, commit=snapshot.commit).data_models
    class_metas = _load_class_metadata(repo)
    docs = _doc_map(repo)
    return DataModelsInputs(class_metas=tuple(class_metas), doc_map=docs)


def compute_data_models_from_inputs(
    inputs: DataModelsInputs,
    snapshot: SnapshotRef,
) -> DataModelsResult:
    """Compute data models from preloaded inputs.

    Returns
    -------
    DataModelsResult
        Container with rows for data_models, data_model_fields,
        and data_model_relationships tables.
    """
    class_metas = list(inputs.class_metas)
    if not class_metas:
        log.info(
            "No class metadata found for %s@%s; returning empty result",
            snapshot.repo,
            snapshot.commit,
        )
        return DataModelsResult(
            model_rows=(),
            field_rows=(),
            relationship_rows=(),
        )

    metas_by_path: dict[str, list[ClassMeta]] = {}
    for meta in class_metas:
        metas_by_path.setdefault(normalize_path(meta.rel_path), []).append(meta)
    docs = inputs.doc_map

    models: list[ModelRecord] = []
    for rel_path, metas in metas_by_path.items():
        abs_path = (Path(snapshot.repo_root) / rel_path).resolve()
        if not abs_path.is_file():
            log.debug("Skipping %s; file missing on disk", abs_path)
            continue
        models.extend(
            _gather_models_for_path(
                rel_path,
                abs_path,
                metas,
                docs,
                snapshot,
            )
        )

    _attach_relationships(models)

    now = datetime.now(tz=UTC)
    model_rows = _build_model_rows(models, snapshot, now)
    field_rows = _build_field_rows(models, snapshot, now)
    relationship_rows = _build_relationship_rows(models, snapshot, now)

    log.info(
        "data_models computed: %d models for %s@%s",
        len(models),
        snapshot.repo,
        snapshot.commit,
    )

    return DataModelsResult(
        model_rows=tuple(model_rows),
        field_rows=tuple(field_rows),
        relationship_rows=tuple(relationship_rows),
    )


def compute_data_models_pure(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> DataModelsResult:
    """Compute data models without writing to database.

    Extract structured data models from class definitions in the snapshot,
    returning structured row data that can be materialized separately.

    Parameters
    ----------
    gateway
        Storage gateway for reading class metadata and docstrings.
    snapshot
        Repository and commit snapshot reference.

    Returns
    -------
    DataModelsResult
        Container with rows for data_models, data_model_fields,
        and data_model_relationships tables.

    Notes
    -----
    This function is a pure transformation that reads from the database but
    does not write. The materialization is handled by the Hamilton native
    module to ensure proper asset catalog tracking.
    """
    inputs = load_data_models_inputs(gateway, snapshot)
    return compute_data_models_from_inputs(inputs, snapshot)


__all__ = [
    "DATA_MODELS_COLS",
    "DATA_MODEL_FIELDS_COLS",
    "DATA_MODEL_RELATIONSHIPS_COLS",
    "DataModelsInputs",
    "DataModelsResult",
    "compute_data_models_from_inputs",
    "compute_data_models_pure",
    "load_data_models_inputs",
]
