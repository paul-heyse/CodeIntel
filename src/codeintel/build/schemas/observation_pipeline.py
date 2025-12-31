"""Schema observation pipeline helpers for build materializers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.schemas.observations import (
    SchemaObservationAccumulator,
    SchemaObservationInputs,
    persist_observation_bundle,
    schema_hints_from_tag_sets,
    table_schema_from_tag_sets,
)
from codeintel.core.schemas.arrow_gen import ExtrasPolicy
from codeintel.storage.duckdb_types import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.build.schemas.observations import SchemaHints
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.gateway.protocol import StorageGateway
    from codeintel.storage.tracking.schema_catalog_models import ParquetStatsPayload

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObservationSetup:
    """Prepared observation configuration for a table."""

    declared_schema: TableSchema | None
    schema_hints: SchemaHints | None
    accumulator: SchemaObservationAccumulator


def build_observation_setup(
    *,
    table_key: str,
    tag_sets: Iterable[Mapping[str, object]],
    declared_schema: TableSchema | None = None,
) -> ObservationSetup:
    """Build observation setup data from schema tags.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    tag_sets
        Iterable of Hamilton tag mappings.
    declared_schema
        Optional declared schema to prefer over tag-derived hints.

    Returns
    -------
    ObservationSetup
        Prepared declared schema, hints, and accumulator.
    """
    resolved_schema = declared_schema or table_schema_from_tag_sets(
        table_key=table_key,
        tag_sets=tag_sets,
    )
    schema_hints = schema_hints_from_tag_sets(tag_sets)
    accumulator = SchemaObservationAccumulator(
        table_key=table_key,
        declared_schema=resolved_schema,
        schema_hints=schema_hints,
    )
    return ObservationSetup(
        declared_schema=resolved_schema,
        schema_hints=schema_hints,
        accumulator=accumulator,
    )


def build_observation_inputs(
    *,
    gateway: StorageGateway,
    table_key: str,
    repo: str | None,
    commit: str | None,
    target_name: str | None,
    extras_policy: ExtrasPolicy | None = None,
    dataset_stats: ParquetStatsPayload | None = None,
    manifest_row_count: int | None = None,
) -> SchemaObservationInputs:
    """Build SchemaObservationInputs with drift history and metadata.

    Parameters
    ----------
    gateway
        Storage gateway used to load drift history.
    table_key
        Fully qualified table key.
    repo
        Repository slug.
    commit
        Commit SHA.
    target_name
        Build target name.
    extras_policy
        Optional extras policy override.
    dataset_stats
        Optional dataset statistics from manifest.
    manifest_row_count
        Optional manifest row count.

    Returns
    -------
    SchemaObservationInputs
        Observation input bundle.
    """
    drift_history = _load_drift_history(gateway=gateway, table_key=table_key)
    return SchemaObservationInputs(
        repo=repo,
        commit=commit,
        target_name=target_name,
        extras_policy=extras_policy,
        drift_history=drift_history,
        dataset_stats=dataset_stats,
        manifest_row_count=manifest_row_count,
    )


def persist_observation(
    *,
    gateway: StorageGateway,
    observation: SchemaObservationAccumulator,
    arrow_schema: pa.Schema,
    inputs: SchemaObservationInputs,
    log: logging.Logger = LOG,
) -> None:
    """Finalize and persist schema observations.

    Parameters
    ----------
    gateway
        Storage gateway for persistence.
    observation
        Accumulator with streamed stats.
    arrow_schema
        Arrow schema observed for the dataset.
    inputs
        Observation inputs (metadata, drift history).
    log
        Logger to record errors.
    """
    try:
        bundle = observation.finalize(arrow_schema=arrow_schema, inputs=inputs)
        persist_observation_bundle(gateway=gateway, bundle=bundle)
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        log.warning(
            "Schema observation persistence failed for %s: %s",
            observation.table_key,
            exc,
        )


def _load_drift_history(
    *,
    gateway: StorageGateway,
    table_key: str,
) -> Sequence[Mapping[str, object] | None] | None:
    try:
        return gateway.schemas.load_recent_drift_summaries(table_key=table_key)
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None


__all__ = [
    "ObservationSetup",
    "build_observation_inputs",
    "build_observation_setup",
    "persist_observation",
]
