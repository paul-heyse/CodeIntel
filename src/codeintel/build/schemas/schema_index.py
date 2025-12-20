"""Schema index derived from the global Hamilton DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.schemas.inference_service import SchemaInferenceService
    from codeintel.build.target_metadata import TargetSystem


SchemaDerivationKind = Literal["inferred_ibis", "contract_override"]


@dataclass(frozen=True, slots=True)
class SchemaDerivation:
    """Describe how a table schema is derived."""

    table_key: str
    kind: SchemaDerivationKind
    source: str
    override_schema: TableSchema | None = None


@dataclass
class SchemaIndex:
    """Resolve table schemas for DAG-produced tables with inference and overrides."""

    derivations: Mapping[str, SchemaDerivation]
    inferable_table_keys: frozenset[str]
    declared_provider: SchemaProvider
    inference_service: SchemaInferenceService
    fallback_to_override_on_error: bool = True
    _cache: dict[str, TableSchema] = field(default_factory=dict, repr=False)

    def get_table_schema(
        self,
        table_key: str,
        *,
        allow_inference: bool = True,
        perform_inference: bool = True,
    ) -> TableSchema | None:
        """Resolve a table schema for a DAG-produced table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).
        allow_inference
            Whether inference is allowed for inferable tables.
        perform_inference
            When False, only return cached inference results or overrides.

        Returns
        -------
        TableSchema | None
            Resolved table schema or None if not found.

        Raises
        ------
        KeyError
            If inference attempts to resolve an unknown table key.
        RuntimeError
            If inference fails due to an unexpected runtime error.
        TypeError
            If inference receives invalid types.
        ValueError
            If inference produces invalid schema data.
        """
        derivation = self.derivations.get(table_key)
        if derivation is None:
            return None

        schema: TableSchema | None = None
        if derivation.kind == "inferred_ibis":
            cached = self._cache.get(table_key)
            if cached is not None:
                schema = cached
            elif not allow_inference or not perform_inference:
                schema = derivation.override_schema
            else:
                try:
                    inferred = self.inference_service.infer_table_schema(
                        table_key,
                        declared_provider=self.declared_provider,
                    )
                except (KeyError, RuntimeError, TypeError, ValueError):
                    if not self.fallback_to_override_on_error:
                        raise
                    schema = derivation.override_schema
                else:
                    self._cache[table_key] = inferred
                    schema = inferred
        elif derivation.override_schema is not None:
            schema = derivation.override_schema
        return schema

    def iter_table_schemas(self, *, allow_inference: bool = True) -> Iterable[TableSchema]:
        """Iterate schemas for all DAG-produced table keys.

        This iteration does not trigger inference; it only yields cached
        inference results or explicit overrides.

        Yields
        ------
        TableSchema
            Resolved table schema for each known table key.
        """
        for table_key in sorted(self.derivations):
            schema = self.get_table_schema(
                table_key,
                allow_inference=allow_inference,
                perform_inference=False,
            )
            if schema is not None:
                yield schema

    def clear_cache(self) -> None:
        """Clear cached inferred schemas."""
        self._cache.clear()


def build_schema_index(
    *,
    system: TargetSystem,
    declared_provider: SchemaProvider,
    inference_service: SchemaInferenceService,
) -> SchemaIndex:
    """Build a SchemaIndex from the global target system.

    Returns
    -------
    SchemaIndex
        Schema index derived from the target system.
    """
    inferable = inference_service.inferable_table_keys(graph=system.graph)
    derivations: dict[str, SchemaDerivation] = {}

    for target in system.graph.all_targets:
        for table_key in target.contract.table_keys:
            override_schema = target.contract.get_table(table_key)
            kind: SchemaDerivationKind = (
                "inferred_ibis" if table_key in inferable else "contract_override"
            )
            derivations[table_key] = SchemaDerivation(
                table_key=table_key,
                kind=kind,
                source=target.name,
                override_schema=override_schema,
            )

    return SchemaIndex(
        derivations=derivations,
        inferable_table_keys=inferable,
        declared_provider=declared_provider,
        inference_service=inference_service,
    )


__all__ = [
    "SchemaDerivation",
    "SchemaDerivationKind",
    "SchemaIndex",
    "build_schema_index",
]
