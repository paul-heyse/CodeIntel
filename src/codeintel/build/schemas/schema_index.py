"""Schema index derived from the global Hamilton DAG."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from codeintel.build.schemas.inference_service import SchemaInferenceService
    from codeintel.build.target_metadata import TargetSystem


SchemaDerivationKind = Literal["explicit_override", "inferred_ibis"]
InferenceStatus = Literal["inferred", "override", "disabled", "error", "pending"]


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
    _inference_errors: dict[str, str] = field(default_factory=dict, repr=False)
    _seed_provider: SchemaProvider | None = field(default=None, repr=False)
    _inference_stack: set[str] = field(default_factory=set, repr=False)

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
        """
        derivation = self.derivations.get(table_key)
        if derivation is None:
            return None

        if derivation.kind != "inferred_ibis":
            return derivation.override_schema
        return self._resolve_inferred_schema(
            table_key,
            derivation=derivation,
            allow_inference=allow_inference,
            perform_inference=perform_inference,
        )

    def iter_table_schemas(self, *, allow_inference: bool = True) -> Iterable[TableSchema]:
        """Iterate schemas for all DAG-produced table keys.

        When inference is enabled, this will infer missing schemas on demand
        and cache the results for subsequent lookups.

        Yields
        ------
        TableSchema
            Resolved table schema for each known table key.
        """
        for table_key in sorted(self.derivations):
            schema = self.get_table_schema(
                table_key,
                allow_inference=allow_inference,
                perform_inference=allow_inference,
            )
            if schema is not None:
                yield schema

    def clear_cache(self) -> None:
        """Clear cached inferred schemas."""
        self._cache.clear()
        self._inference_errors.clear()

    def prefill_cache(self, schemas: Mapping[str, TableSchema]) -> None:
        """Prefill the inference cache with known schemas.

        Parameters
        ----------
        schemas
            Mapping of table_key to schema to seed into the cache.
        """
        if not schemas:
            return
        self._cache.update(schemas)
        for table_key in schemas:
            self._clear_inference_error(table_key)

    def get_inference_error(self, table_key: str) -> str | None:
        """Return the most recent inference error for a table key.

        Returns
        -------
        str | None
            Error message when inference failed for the table key.
        """
        return self._inference_errors.get(table_key)

    def iter_inference_errors(self) -> Iterable[tuple[str, str]]:
        """Iterate inference errors in deterministic order.

        Yields
        ------
        tuple[str, str]
            Table key and error message.
        """
        for table_key in sorted(self._inference_errors):
            yield table_key, self._inference_errors[table_key]

    def inference_status_for(
        self,
        table_key: str,
        *,
        allow_inference: bool | None = None,
    ) -> InferenceStatus | None:
        """Return inference status for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).
        allow_inference
            Whether inference is enabled for this lookup. When None, defaults
            to True.

        Returns
        -------
        InferenceStatus | None
            Inference status when table_key is known; otherwise None.
        """
        derivation = self.derivations.get(table_key)
        if derivation is None:
            return None
        if derivation.kind != "inferred_ibis":
            return "override"
        if table_key in self._cache:
            return "inferred"
        if self.get_inference_error(table_key) is not None:
            return "error"
        if allow_inference is False:
            return "disabled"
        return "pending"

    def raise_if_inference_recursive(self, table_key: str) -> None:
        """Raise if the table is already being inferred to prevent recursion.

        Raises
        ------
        RuntimeError
            If inference for the table key is already in progress.
        """
        if table_key in self._inference_stack:
            msg = f"Recursive schema inference detected for {table_key}"
            raise RuntimeError(msg)

    def _resolve_inferred_schema(
        self,
        table_key: str,
        *,
        derivation: SchemaDerivation,
        allow_inference: bool,
        perform_inference: bool,
    ) -> TableSchema | None:
        cached = self._cache.get(table_key)
        if cached is not None:
            return cached
        if not allow_inference or not perform_inference:
            return derivation.override_schema
        return self._infer_and_cache_schema(table_key, derivation=derivation)

    def _infer_and_cache_schema(
        self,
        table_key: str,
        *,
        derivation: SchemaDerivation,
    ) -> TableSchema | None:
        with self._inference_guard(table_key):
            return self._infer_with_fallback(table_key, derivation=derivation)

    def _infer_with_fallback(
        self,
        table_key: str,
        *,
        derivation: SchemaDerivation,
    ) -> TableSchema | None:
        try:
            inferred = self.inference_service.infer_table_schema(
                table_key,
                declared_provider=self._schema_seed_provider(),
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._record_inference_error(table_key, exc)
            override_schema = derivation.override_schema
            if override_schema is None:
                return None
            if not self.fallback_to_override_on_error:
                raise
            return override_schema

        self._clear_inference_error(table_key)
        self._cache[table_key] = inferred
        return inferred

    @contextmanager
    def _inference_guard(self, table_key: str) -> Iterator[None]:
        self.raise_if_inference_recursive(table_key)
        self._inference_stack.add(table_key)
        try:
            yield
        finally:
            self._inference_stack.remove(table_key)

    def _record_inference_error(self, table_key: str, exc: Exception) -> None:
        detail = str(exc)
        label = type(exc).__name__
        message = f"{label}: {detail}" if detail else label
        self._inference_errors[table_key] = message

    def _clear_inference_error(self, table_key: str) -> None:
        self._inference_errors.pop(table_key, None)

    def _schema_seed_provider(self) -> SchemaProvider:
        if self._seed_provider is None:
            self._seed_provider = _SchemaIndexSeedProvider(
                declared_provider=self.declared_provider,
                schema_index=self,
            )
        return self._seed_provider


@dataclass(frozen=True, slots=True)
class _SchemaIndexSeedProvider:
    declared_provider: SchemaProvider
    schema_index: SchemaIndex

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        self.schema_index.raise_if_inference_recursive(table_key)
        schema = self.declared_provider.get_table_schema(table_key)
        if schema is not None:
            return schema
        return self.schema_index.get_table_schema(
            table_key,
            allow_inference=True,
            perform_inference=True,
        )

    def require_table_schema(self, table_key: str) -> TableSchema:
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        return self.schema_index.iter_table_schemas(allow_inference=True)


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

    Raises
    ------
    ValueError
        If non-inferable outputs are missing explicit overrides.
    """
    inferable = inference_service.inferable_table_keys(graph=system.graph)
    derivations: dict[str, SchemaDerivation] = {}
    missing_overrides: list[tuple[str, str]] = []

    for target in system.graph.all_targets:
        for table_key in target.contract.table_keys:
            override_schema = target.contract.get_table(table_key)
            if table_key in inferable:
                kind: SchemaDerivationKind = "inferred_ibis"
            else:
                if override_schema is None:
                    missing_overrides.append((table_key, target.name))
                    continue
                kind = "explicit_override"
            derivations[table_key] = SchemaDerivation(
                table_key=table_key,
                kind=kind,
                source=target.name,
                override_schema=override_schema,
            )

    if missing_overrides:
        missing = ", ".join(
            f"{table_key} (target={target_name})"
            for table_key, target_name in sorted(missing_overrides)
        )
        msg = f"Missing explicit schema overrides for non-inferable outputs: {missing}"
        raise ValueError(msg)

    return SchemaIndex(
        derivations=derivations,
        inferable_table_keys=inferable,
        declared_provider=declared_provider,
        inference_service=inference_service,
    )


__all__ = [
    "InferenceStatus",
    "SchemaDerivation",
    "SchemaDerivationKind",
    "SchemaIndex",
    "build_schema_index",
]
