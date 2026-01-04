"""Pure compute functions for data model extraction.

This module provides pure compute functions that return row data without
performing any database writes. The materialization is handled by the
Hamilton native module in `build/hamilton/native/analytics/data_models.py`.

The functions extract structured data models from class definitions,
returning structured result containers that can be materialized to DuckDB
tables by the build system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.row_builders import rows_to_tuples_for_table
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

if TYPE_CHECKING:
    import pyarrow as pa

    from codeintel.build.analytics.data_models.core import ClassMeta, ModelRecord
    from codeintel.config.primitives import SnapshotRef


log = logging.getLogger(__name__)

DATA_MODELS_TABLE_KEY = "analytics.data_models"
DATA_MODEL_FIELDS_TABLE_KEY = "analytics.data_model_fields"
DATA_MODEL_RELATIONSHIPS_TABLE_KEY = "analytics.data_model_relationships"


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
    rows = [
        {
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "model_id": model.model_id,
            "goid_h128": model.goid,
            "model_name": model.model_name,
            "module": model.module,
            "rel_path": model.rel_path,
            "model_kind": model.model_kind,
            "base_classes_json": list(model.base_classes),
            "doc_short": model.doc_short,
            "doc_long": model.doc_long,
            "created_at": now,
        }
        for model in models
    ]
    return rows_to_tuples_for_table(DATA_MODELS_TABLE_KEY, rows)


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
    rows = [
        {
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "model_id": model.model_id,
            "field_name": field_spec.name,
            "field_type": field_spec.type,
            "required": field_spec.required,
            "has_default": field_spec.has_default,
            "default_expr": field_spec.default_expr,
            "constraints_json": dict(field_spec.constraints),
            "source": field_spec.source,
            "rel_path": model.rel_path,
            "lineno": field_spec.lineno,
            "created_at": now,
        }
        for model in models
        for field_spec in model.fields
    ]
    return rows_to_tuples_for_table(DATA_MODEL_FIELDS_TABLE_KEY, rows)


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
    rows = [
        {
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "source_model_id": model.model_id,
            "target_model_id": rel.target_model_id,
            "target_module": rel.target_module,
            "target_model_name": rel.target_model_name,
            "field_name": rel.field_name,
            "relationship_kind": rel.kind,
            "multiplicity": rel.multiplicity,
            "via": rel.via,
            "evidence_json": rel.evidence if rel.evidence else None,
            "rel_path": rel.rel_path,
            "lineno": rel.lineno,
            "created_at": now,
        }
        for model in models
        for rel in model.relationships
    ]
    return rows_to_tuples_for_table(DATA_MODEL_RELATIONSHIPS_TABLE_KEY, rows)


def load_data_models_inputs(
    snapshot: SnapshotRef,
    *,
    goids_frame: pa.Table,
    modules_frame: pa.Table,
    docstrings_frame: pa.Table,
) -> DataModelsInputs:
    """Load tabular inputs required for data model computation.

    Returns
    -------
    DataModelsInputs
        Prepared class metadata and docstring map for the snapshot.
    """
    class_metas = _load_class_metadata(
        goids_frame,
        modules_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    docs = _doc_map(docstrings_frame, repo=snapshot.repo, commit=snapshot.commit)
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
    snapshot: SnapshotRef,
    *,
    goids_frame: pa.Table,
    modules_frame: pa.Table,
    docstrings_frame: pa.Table,
) -> DataModelsResult:
    """Compute data models without writing to database.

    Extract structured data models from class definitions in the snapshot,
    returning structured row data that can be materialized separately.

    Parameters
    ----------
    snapshot
        Repository and commit snapshot reference.
    goids_frame
        GOID rows for the snapshot.
    modules_frame
        Module rows for the snapshot.
    docstrings_frame
        Docstring rows for the snapshot.

    Returns
    -------
    DataModelsResult
        Container with rows for data_models, data_model_fields,
        and data_model_relationships tables.

    Notes
    -----
    This function is a pure transformation that reads from tabular inputs but
    does not write. The materialization is handled by the Hamilton native
    module to ensure proper asset catalog tracking.
    """
    inputs = load_data_models_inputs(
        snapshot,
        goids_frame=goids_frame,
        modules_frame=modules_frame,
        docstrings_frame=docstrings_frame,
    )
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
