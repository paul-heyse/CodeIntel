"""Contract bundle aggregation for schema artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import msgspec

from codeintel.core.columnar.schema_alignment import extras_policy_from_schema
from codeintel.core.schemas.arrow_gen import ExtrasPolicy
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.row_models import (
    GeneratedRowBinding,
    RowSerializer,
    RowStructBuilder,
    RowStructSerializer,
)
from codeintel.core.validation.pandera_schema import (
    pandera_schema_for_table,
    resolve_extras_policy,
)
from codeintel.core.validation.profiles import ValidationProfile

if TYPE_CHECKING:
    import pyarrow as pa

    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord


@dataclass(frozen=True, slots=True)
class ContractBundle:
    """Unified bundle of schema artifacts for a table key.

    Attributes
    ----------
    table_key
        Fully qualified table key (schema.table).
    table_schema
        Resolved table schema, if available.
    arrow_schema
        Arrow schema with metadata, when resolved.
    json_schema
        JSON Schema payload, if available.
    json_schema_id
        JSON Schema identifier for the table key.
    json_schema_digest
        Stable digest for the JSON Schema payload.
    row_binding
        Generated row binding for the table schema.
    schema_hash
        Schema hash for cache invalidation.
    pandera_schema
        Optional Pandera schema instance for validation.
    """

    table_key: str
    table_schema: TableSchema | None
    arrow_schema: pa.Schema | None
    json_schema: dict[str, Any] | None
    json_schema_id: str | None
    json_schema_digest: str | None
    row_binding: GeneratedRowBinding | None
    schema_hash: str | None
    pandera_schema: object | None = None

    @property
    def row_struct(self) -> type[msgspec.Struct] | None:
        """Return the msgspec struct row model, if available."""
        if self.row_binding is None:
            return None
        return self.row_binding.struct_model

    @property
    def row_struct_builder(self) -> RowStructBuilder | None:
        """Return the row struct builder, if available."""
        if self.row_binding is None:
            return None
        return self.row_binding.struct_builder

    @property
    def row_serializer(self) -> RowSerializer | None:
        """Return the mapping-based row serializer, if available."""
        if self.row_binding is None:
            return None
        return self.row_binding.serializer

    @property
    def row_struct_serializer(self) -> RowStructSerializer | None:
        """Return the struct-based row serializer, if available."""
        if self.row_binding is None:
            return None
        return self.row_binding.struct_serializer

    def with_pandera_schema(
        self,
        *,
        observation: SchemaObservationRecord | None = None,
        extras_policy: ExtrasPolicy | None = None,
        validation_profile: ValidationProfile | None = None,
    ) -> ContractBundle:
        """Return a copy of the bundle with a Pandera schema attached.

        Parameters
        ----------
        observation
            Optional schema observation for derived validation settings.
        extras_policy
            Explicit extras policy override when available.
        validation_profile
            Validation profile controlling schema-only vs data checks.

        Returns
        -------
        ContractBundle
            Updated bundle containing a Pandera schema when possible.
        """
        if self.table_schema is None:
            return self
        if self.pandera_schema is not None:
            return self
        resolved_policy = extras_policy
        if observation is not None:
            resolved_policy = resolve_extras_policy(observation, fallback=resolved_policy)
        if resolved_policy is None and self.arrow_schema is not None:
            resolved_policy = extras_policy_from_schema(self.arrow_schema)
        pandera_schema = pandera_schema_for_table(
            table_schema=self.table_schema,
            observation=observation,
            extras_policy=resolved_policy,
            validation_profile=validation_profile,
        )
        return replace(self, pandera_schema=pandera_schema)


__all__ = [
    "ContractBundle",
]
