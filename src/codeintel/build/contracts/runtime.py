"""Runtime resolution for lazy contract references."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from codeintel.build.contracts.ref import ContractRef
from codeintel.build.contracts.registry import ContractResolver
from codeintel.build.contracts.types import (
    UNSET,
    ContractOverrides,
    ContractPolicy,
    TableContractSpec,
    UnsetType,
)
from codeintel.core.schemas import SchemaService, get_schema_service
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


@dataclass(frozen=True, slots=True)
class _ContractPolicyKey:
    extras_policy: object | None
    validation_profile: object | None
    coerce_types: bool
    allow_nulls: bool


@dataclass(frozen=True, slots=True)
class _ContractOverridesKey:
    ops_module: ModuleType | None | UnsetType
    columns_to_pass: tuple[str, ...] | UnsetType
    required_cols: tuple[str, ...] | UnsetType
    clip_column: str | None | UnsetType
    policy: _ContractPolicyKey | UnsetType


@dataclass(frozen=True, slots=True)
class _ContractResolutionKey:
    table_key: str
    target_name: str
    input_name: str
    overrides: _ContractOverridesKey


@dataclass(slots=True)
class ContractRuntime:
    """Resolve ContractRef instances into contract specs."""

    schema_service: SchemaService
    resolver: ContractResolver = field(default_factory=ContractResolver)
    _cache: dict[_ContractResolutionKey, TableContractSpec] = field(default_factory=dict)

    def resolve(self, ref: ContractRef) -> TableContractSpec:
        """Resolve a contract reference into a concrete spec.

        Returns
        -------
        TableContractSpec
            Resolved contract spec derived from schema and overrides.
        """
        resolved_overrides = _resolved_overrides(ref)
        cache_key = _resolution_key(ref, resolved_overrides)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        table_schema = self.schema_service.require_table_schema(ref.table_key)
        spec = self.resolver.resolve(
            table_schema=table_schema,
            domain=ref.domain,
            target=ref.target_name,
            overrides=resolved_overrides,
        )
        self._cache[cache_key] = spec
        return spec


@dataclass(slots=True)
class _ContractRuntimeState:
    runtime: ContractRuntime | None = None
    schema_service_id: int | None = None


_CONTRACT_RUNTIME_STATE = _ContractRuntimeState()


def configure_contract_runtime(*, schema_service: SchemaService) -> ContractRuntime:
    """Configure the global contract runtime for a schema service.

    Returns
    -------
    ContractRuntime
        Configured contract runtime for resolving contract refs.
    """
    state = _CONTRACT_RUNTIME_STATE
    if state.runtime is not None and state.schema_service_id == id(schema_service):
        return state.runtime
    runtime = ContractRuntime(schema_service=schema_service)
    state.runtime = runtime
    state.schema_service_id = id(schema_service)
    return runtime


def get_contract_runtime() -> ContractRuntime:
    """Return the configured contract runtime.

    Returns
    -------
    ContractRuntime
        Runtime used to resolve ContractRef instances.
    """
    state = _CONTRACT_RUNTIME_STATE
    if state.runtime is not None:
        return state.runtime
    schema_service = _fallback_schema_service()
    return configure_contract_runtime(schema_service=schema_service)


def _fallback_schema_service() -> SchemaService:
    try:
        return get_schema_service()
    except RuntimeError:
        provider = MappingSchemaProvider(dict(TABLE_SCHEMAS))
        return SchemaService(table_provider=provider)


def _resolved_overrides(ref: ContractRef) -> ContractOverrides:
    overrides = ref.overrides or ContractOverrides()
    if overrides.input_name is UNSET:
        return replace(overrides, input_name=ref.input_name)
    if overrides.input_name != ref.input_name:
        msg = (
            "ContractRef input_name mismatch: "
            f"{overrides.input_name!r} != {ref.input_name!r}"
        )
        raise ValueError(msg)
    return overrides


def _resolution_key(
    ref: ContractRef,
    overrides: ContractOverrides,
) -> _ContractResolutionKey:
    return _ContractResolutionKey(
        table_key=ref.table_key,
        target_name=ref.target_name,
        input_name=ref.input_name,
        overrides=_overrides_key(overrides),
    )


def _overrides_key(overrides: ContractOverrides) -> _ContractOverridesKey:
    return _ContractOverridesKey(
        ops_module=overrides.ops_module,
        columns_to_pass=_sequence_key(overrides.columns_to_pass),
        required_cols=_sequence_key(overrides.required_cols),
        clip_column=overrides.clip_column,
        policy=_policy_key(overrides.policy),
    )


def _sequence_key(value: Sequence[str] | UnsetType) -> tuple[str, ...] | UnsetType:
    if value is UNSET:
        return UNSET
    return tuple(value)


def _policy_key(value: ContractPolicy | UnsetType) -> _ContractPolicyKey | UnsetType:
    if value is UNSET:
        return UNSET
    return _ContractPolicyKey(
        extras_policy=value.extras_policy,
        validation_profile=value.validation_profile,
        coerce_types=value.coerce_types,
        allow_nulls=value.allow_nulls,
    )


__all__ = ["ContractRuntime", "configure_contract_runtime", "get_contract_runtime"]
