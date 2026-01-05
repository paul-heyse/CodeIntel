"""Schema observation pipeline helpers for build materializers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.schemas.observations import (
    SchemaObservationAccumulator,
    SchemaObservationInputs,
    persist_observation_bundle,
    schema_hints_from_tag_sets,
    table_schema_from_tag_sets,
)
from codeintel.core.duckdb_types import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.build.meta.bundle import BuildMetadataBundleWriter
    from codeintel.build.schemas.observations import SchemaHints
    from codeintel.core.gateway import BuildGateway
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObservationSetup:
    """Prepared observation configuration for a table."""

    declared_schema: TableSchema | None
    schema_hints: SchemaHints | None
    accumulator: SchemaObservationAccumulator


@dataclass(frozen=True, slots=True)
class ObservationPersistContext:
    """Persistence context for schema observations."""

    gateway: BuildGateway | None
    metadata_bundle: BuildMetadataBundleWriter | None
    log: logging.Logger = LOG


@dataclass(frozen=True, slots=True)
class ObservationPersistPayload:
    """Payload required to persist a schema observation."""

    observation: SchemaObservationAccumulator
    arrow_schema: pa.Schema
    inputs: SchemaObservationInputs


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
    gateway: BuildGateway | None,
    table_key: str,
    base: SchemaObservationInputs,
) -> SchemaObservationInputs:
    """Build SchemaObservationInputs with drift history and metadata.

    Parameters
    ----------
    gateway
        Storage gateway used to load drift history.
    table_key
        Fully qualified table key.
    base
        Base inputs with repo/commit/target metadata.

    Returns
    -------
    SchemaObservationInputs
        Observation input bundle.
    """
    if gateway is None:
        return base
    drift_history = base.drift_history
    if drift_history is None:
        drift_history = _load_drift_history(gateway=gateway, table_key=table_key)
    previous = base.previous_observation
    if previous is None:
        previous = _load_latest_observation(gateway=gateway, table_key=table_key)
    if drift_history is base.drift_history and previous is base.previous_observation:
        return base
    return replace(
        base,
        drift_history=drift_history,
        previous_observation=previous,
    )


def persist_observation(
    *,
    context: ObservationPersistContext,
    payload: ObservationPersistPayload,
) -> None:
    """Finalize and persist schema observations.

    Parameters
    ----------
    context
        Persistence context (gateway, metadata bundle, logger).
    payload
        Observation payload with schema, inputs, and accumulator.
    """
    try:
        bundle = payload.observation.finalize(
            arrow_schema=payload.arrow_schema,
            inputs=payload.inputs,
        )
        persist_observation_bundle(
            bundle=bundle,
            metadata_bundle=context.metadata_bundle,
            gateway=context.gateway,
        )
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        context.log.warning(
            "Schema observation persistence failed for %s: %s",
            payload.observation.table_key,
            exc,
        )


def _load_drift_history(
    *,
    gateway: BuildGateway,
    table_key: str,
) -> Sequence[Mapping[str, object] | None] | None:
    try:
        return gateway.schemas.load_recent_drift_summaries(table_key=table_key)
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None


def _load_latest_observation(
    *,
    gateway: BuildGateway,
    table_key: str,
) -> SchemaObservationRecord | None:
    try:
        return gateway.schemas.load_latest_schema_observation(table_key=table_key)
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None


__all__ = [
    "ObservationSetup",
    "build_observation_inputs",
    "build_observation_setup",
    "persist_observation",
]
