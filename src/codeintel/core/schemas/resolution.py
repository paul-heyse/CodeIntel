"""Canonical schema resolution helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.core.columnar.ipc import schema_from_ipc_payload
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.authority import SchemaDerivation
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.core.schemas.service import SchemaService, get_schema_service

if TYPE_CHECKING:
    import pyarrow as pa

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord


@runtime_checkable
class SchemaObservationProvider(Protocol):
    """Protocol for loading schema observations."""

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None:
        """Return the latest observation for a table key."""
        ...


@runtime_checkable
class SchemaDerivationProvider(Protocol):
    """Protocol for schema providers with derivation metadata."""

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        """Return derivation metadata for a table key."""
        ...


class SchemaResolutionSource(Enum):
    """Schema resolution source classification."""

    OBSERVED = "observed"
    OVERRIDE = "override"
    INFERRED = "inferred"
    DECLARED = "declared"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class SchemaResolutionResult:
    """Resolved schema record with provenance."""

    table_key: str
    table_schema: TableSchema | None
    observation: SchemaObservationRecord | None
    source: SchemaResolutionSource


@dataclass(frozen=True, slots=True)
class ResolvedSchemaProvider:
    """Schema provider wrapper that prefers observations."""

    observation_provider: SchemaObservationProvider | None
    fallback_provider: SchemaProvider

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return the resolved schema for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Resolved table schema when available.
        """
        result = resolve_table_schema(
            table_key,
            observation_provider=self.observation_provider,
            schema_provider=self.fallback_provider,
        )
        return result.table_schema

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return the resolved schema for a table key, raising if missing.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Resolved table schema.

        Raises
        ------
        KeyError
            If the table key cannot be resolved.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate resolved schemas in fallback provider order.

        Yields
        ------
        TableSchema
            Resolved table schemas.
        """
        for schema in self.fallback_provider.iter_table_schemas():
            resolved = resolve_table_schema(
                schema.table_key,
                observation_provider=self.observation_provider,
                schema_provider=self.fallback_provider,
            )
            if resolved.table_schema is not None:
                yield resolved.table_schema

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        """Return derivation metadata for resolved schemas.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        SchemaDerivation | None
            Derivation metadata when available.
        """
        result = resolve_table_schema(
            table_key,
            observation_provider=self.observation_provider,
            schema_provider=self.fallback_provider,
        )
        if result.source == SchemaResolutionSource.OBSERVED and result.table_schema is not None:
            return SchemaDerivation(
                table_key=table_key,
                source_kind="observed",
                source_ref="observation",
                schema_hash=schema_hash(result.table_schema),
            )
        if isinstance(self.fallback_provider, SchemaDerivationProvider):
            return self.fallback_provider.derivation(table_key)
        return None


@dataclass(frozen=True, slots=True)
class ResolvedArrowSchemaProvider:
    """Arrow schema provider wrapper that prefers observations."""

    observation_provider: SchemaObservationProvider | None
    fallback_provider: SchemaProvider

    def get_arrow_schema(self, table_key: str) -> pa.Schema | None:
        """Return the resolved Arrow schema for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        pa.Schema | None
            Resolved Arrow schema when available.
        """
        return resolve_arrow_schema(
            table_key,
            observation_provider=self.observation_provider,
            schema_provider=self.fallback_provider,
        )


def resolve_table_schema(
    table_key: str,
    *,
    observation_provider: SchemaObservationProvider | None = None,
    schema_provider: SchemaProvider | None = None,
) -> SchemaResolutionResult:
    """Resolve a table schema using observed-first precedence.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    observation_provider
        Optional provider for schema observations.
    schema_provider
        Optional schema provider. Falls back to the configured SchemaService.

    Returns
    -------
    SchemaResolutionResult
        Resolved schema record and provenance.
    """
    observation = _load_observation(
        table_key=table_key,
        observation_provider=observation_provider,
    )
    observed_schema = (
        _table_schema_from_observation(table_key, observation) if observation else None
    )
    if observed_schema is not None:
        return SchemaResolutionResult(
            table_key=table_key,
            table_schema=observed_schema,
            observation=observation,
            source=SchemaResolutionSource.OBSERVED,
        )

    provider = schema_provider or _schema_provider_from_service()
    if provider is None:
        return SchemaResolutionResult(
            table_key=table_key,
            table_schema=None,
            observation=observation,
            source=SchemaResolutionSource.MISSING,
        )
    table_schema = provider.get_table_schema(table_key)
    if table_schema is None:
        return SchemaResolutionResult(
            table_key=table_key,
            table_schema=None,
            observation=observation,
            source=SchemaResolutionSource.MISSING,
        )
    source = _source_from_provider(provider, table_key)
    return SchemaResolutionResult(
        table_key=table_key,
        table_schema=table_schema,
        observation=observation,
        source=source,
    )


def resolve_arrow_schema(
    table_key: str,
    *,
    observation_provider: SchemaObservationProvider | None = None,
    schema_provider: SchemaProvider | None = None,
) -> pa.Schema | None:
    """Resolve an Arrow schema using observed-first precedence.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    observation_provider
        Optional provider for schema observations.
    schema_provider
        Optional schema provider. Falls back to the configured SchemaService.

    Returns
    -------
    pyarrow.Schema | None
        Resolved Arrow schema when available.
    """
    result = resolve_table_schema(
        table_key,
        observation_provider=observation_provider,
        schema_provider=schema_provider,
    )
    if result.observation is not None:
        observed_schema = _arrow_schema_from_observation(result.observation)
        if observed_schema is not None:
            return observed_schema
    if result.table_schema is None:
        return None
    return arrow_schema_from_table_schema(table_schema=result.table_schema)


def _load_observation(
    *,
    table_key: str,
    observation_provider: SchemaObservationProvider | None,
) -> SchemaObservationRecord | None:
    if observation_provider is None:
        return None
    try:
        return observation_provider.load_latest_schema_observation(table_key=table_key)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _table_schema_from_observation(
    table_key: str,
    observation: SchemaObservationRecord,
) -> TableSchema | None:
    arrow_schema = _arrow_schema_from_observation(observation)
    if arrow_schema is None:
        return None
    return table_schema_from_arrow_schema(arrow_schema=arrow_schema, table_key=table_key)


def _arrow_schema_from_observation(observation: SchemaObservationRecord) -> pa.Schema | None:
    return schema_from_ipc_payload(observation.arrow_schema_ipc_b64)


def _schema_provider_from_service() -> SchemaProvider | None:
    service = _schema_service()
    if service is None:
        return None
    return service.table_provider


def _schema_service() -> SchemaService | None:
    try:
        return get_schema_service()
    except RuntimeError:
        return None


def _source_from_provider(provider: SchemaProvider, table_key: str) -> SchemaResolutionSource:
    if isinstance(provider, SchemaDerivationProvider):
        derivation = provider.derivation(table_key)
        if derivation is None:
            return SchemaResolutionSource.MISSING
        return _source_from_kind(derivation.source_kind)
    return SchemaResolutionSource.DECLARED


def _source_from_kind(kind: str) -> SchemaResolutionSource:
    if kind in {"explicit_override", "override"}:
        return SchemaResolutionSource.OVERRIDE
    if kind in {"inferred_relation", "inferred"}:
        return SchemaResolutionSource.INFERRED
    if kind in {"declared_source", "declared"}:
        return SchemaResolutionSource.DECLARED
    return SchemaResolutionSource.DECLARED


__all__ = [
    "ResolvedArrowSchemaProvider",
    "ResolvedSchemaProvider",
    "SchemaDerivationProvider",
    "SchemaObservationProvider",
    "SchemaResolutionResult",
    "SchemaResolutionSource",
    "resolve_arrow_schema",
    "resolve_table_schema",
]
