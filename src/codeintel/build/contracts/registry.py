"""Registry helpers for build-time table contract resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Protocol, TypedDict, TypeVar, Unpack, cast

from codeintel.build.contracts.types import (
    UNSET,
    ContractDescriptor,
    ContractOverrides,
    ContractPolicy,
    TableContractSpec,
    UnsetType,
)
from codeintel.build.schemas import get_contract_for_table_key
from codeintel.build.schemas.service import get_schema_service
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema

_T = TypeVar("_T")


class ContractRegistry(Protocol):
    """Protocol for table contract registries."""

    def get_contract(
        self,
        *,
        table_key: str,
        domain: str,
        target: str,
        overrides: ContractOverrides | None = None,
    ) -> TableContractSpec | None:
        """Return a table contract spec, or None if it cannot be resolved."""
        ...

    def require_contract(
        self,
        *,
        table_key: str,
        domain: str,
        target: str,
        overrides: ContractOverrides | None = None,
    ) -> TableContractSpec:
        """Return a table contract spec, raising if it cannot be resolved."""
        ...


def _override_value(value: _T | UnsetType, default: _T) -> _T:
    if value is UNSET:
        return default
    return cast("_T", value)


def _override_sequence(
    value: Sequence[str] | UnsetType,
    default: Sequence[str],
) -> tuple[str, ...]:
    if value is UNSET:
        return tuple(default)
    return tuple(cast("Sequence[str]", value))


def _validate_domain(table_schema: TableSchema, domain: str) -> None:
    if table_schema.schema != domain:
        msg = (
            "Contract domain mismatch for "
            f"{table_schema.table_key}: expected {table_schema.schema!r}, got {domain!r}"
        )
        raise ValueError(msg)


def _domain_from_table_key(table_key: str) -> str:
    if "." not in table_key:
        msg = f"Invalid table key: {table_key!r}"
        raise ValueError(msg)
    domain, table_name = table_key.split(".", maxsplit=1)
    if not domain or not table_name:
        msg = f"Invalid table key: {table_key!r}"
        raise ValueError(msg)
    return domain


@dataclass(frozen=True, slots=True)
class ContractResolver:
    """Resolve contract specs from schema defaults and overrides."""

    default_policy: ContractPolicy = field(default_factory=ContractPolicy)

    def resolve(
        self,
        *,
        table_schema: TableSchema,
        domain: str,
        target: str,
        overrides: ContractOverrides | None = None,
    ) -> TableContractSpec:
        """Return a contract spec derived from schema defaults and overrides.

        Returns
        -------
        TableContractSpec
            Resolved contract spec derived from the table schema.
        """
        _validate_domain(table_schema, domain)
        descriptor = contract_descriptor_for_table_schema(table_schema)
        spec = TableContractSpec(
            table_key=table_schema.table_key,
            domain=domain,
            target=target,
            ops_module=None,
            columns_to_pass=(),
            policy=self.default_policy,
            contract_version=descriptor.contract_version,
            contract_hash=descriptor.contract_hash,
        )
        return _apply_overrides(spec, overrides)


@dataclass(frozen=True, slots=True)
class ContractForTableInput:
    """Specification for building a contract spec for a target table."""

    table_key: str
    input_name: str
    ops_module: ModuleType | UnsetType | None = UNSET
    columns_to_pass: Sequence[str] | UnsetType = UNSET
    required_cols: Sequence[str] | UnsetType = UNSET
    clip_column: str | UnsetType | None = UNSET
    policy: ContractPolicy | UnsetType = UNSET


class ContractForTableOverrides(TypedDict, total=False):
    """Override values accepted by contract_for_table."""

    ops_module: ModuleType | None
    columns_to_pass: Sequence[str]
    required_cols: Sequence[str]
    clip_column: str | None
    policy: ContractPolicy


def _resolve_contract_version(
    *,
    table_key: str,
) -> str | None:
    try:
        contract = get_contract_for_table_key(table_key)
    except (KeyError, ValueError, RuntimeError):
        return None
    return contract.schema_version


def contract_descriptor_for_table_schema(table_schema: TableSchema) -> ContractDescriptor:
    """Return contract identity metadata for a table schema.

    Returns
    -------
    ContractDescriptor
        Contract descriptor with version and hash values.
    """
    schema_hash_value = schema_hash(table_schema)
    return ContractDescriptor(
        table_key=table_schema.table_key,
        contract_version=_resolve_contract_version(table_key=table_schema.table_key),
        contract_hash=schema_hash_value,
    )


def contract_descriptor_for_table_key(table_key: str) -> ContractDescriptor | None:
    """Return contract identity metadata for a table key.

    Returns
    -------
    ContractDescriptor | None
        Contract descriptor when table schema exists, otherwise None.
    """
    schema_service = get_schema_service()
    table_schema = schema_service.get_table_schema(table_key)
    if table_schema is None:
        return None
    return contract_descriptor_for_table_schema(table_schema)


def _apply_overrides(
    spec: TableContractSpec,
    overrides: ContractOverrides | None,
) -> TableContractSpec:
    if overrides is None:
        return spec
    return replace(
        spec,
        input_name=_override_value(overrides.input_name, spec.input_name),
        ops_module=_override_value(overrides.ops_module, spec.ops_module),
        columns_to_pass=_override_sequence(overrides.columns_to_pass, spec.columns_to_pass),
        required_cols=_override_sequence(overrides.required_cols, spec.required_cols),
        clip_column=_override_value(overrides.clip_column, spec.clip_column),
        policy=_override_value(overrides.policy, spec.policy),
    )


@dataclass(frozen=True, slots=True)
class SchemaBackedContractRegistry:
    """Resolve contract specs backed by the schema service."""

    resolver: ContractResolver = field(default_factory=ContractResolver)

    def get_contract(
        self,
        *,
        table_key: str,
        domain: str,
        target: str,
        overrides: ContractOverrides | None = None,
    ) -> TableContractSpec | None:
        """Return a table contract spec or None when missing.

        Returns
        -------
        TableContractSpec | None
            Contract spec when available, otherwise None.
        """
        schema_service = get_schema_service()
        table_schema = schema_service.get_table_schema(table_key)
        if table_schema is None:
            return None
        return self.resolver.resolve(
            table_schema=table_schema,
            domain=domain,
            target=target,
            overrides=overrides,
        )

    def require_contract(
        self,
        *,
        table_key: str,
        domain: str,
        target: str,
        overrides: ContractOverrides | None = None,
    ) -> TableContractSpec:
        """Return a table contract spec, raising when missing.

        Returns
        -------
        TableContractSpec
            Resolved contract spec.
        """
        schema_service = get_schema_service()
        table_schema = schema_service.require_table_schema(table_key)
        return self.resolver.resolve(
            table_schema=table_schema,
            domain=domain,
            target=target,
            overrides=overrides,
        )


_CONTRACT_REGISTRY_STATE: dict[str, ContractRegistry | None] = {"registry": None}


def set_contract_registry(*, registry: ContractRegistry | None) -> None:
    """Set the global contract registry instance."""
    _CONTRACT_REGISTRY_STATE["registry"] = registry


def get_contract_registry() -> ContractRegistry:
    """Return the configured contract registry.

    Returns
    -------
    ContractRegistry
        Registry instance used for contract resolution.
    """
    registry = _CONTRACT_REGISTRY_STATE["registry"]
    if registry is None:
        registry = SchemaBackedContractRegistry()
        _CONTRACT_REGISTRY_STATE["registry"] = registry
    return registry


def get_contract(
    *,
    table_key: str,
    domain: str,
    target: str,
    overrides: ContractOverrides | None = None,
) -> TableContractSpec | None:
    """Return a table contract spec, or None if it cannot be resolved.

    Returns
    -------
    TableContractSpec | None
        Contract spec when available, otherwise None.
    """
    registry = get_contract_registry()
    return registry.get_contract(
        table_key=table_key,
        domain=domain,
        target=target,
        overrides=overrides,
    )


def get_contract_for_target(
    *,
    table_key: str,
    target_name: str,
    overrides: ContractOverrides | None = None,
) -> TableContractSpec | None:
    """Return a table contract spec for a target.

    Returns
    -------
    TableContractSpec | None
        Contract spec when available, otherwise None.
    """
    domain = _domain_from_table_key(table_key)
    return get_contract(
        table_key=table_key,
        domain=domain,
        target=target_name,
        overrides=overrides,
    )


def require_contract(
    *,
    table_key: str,
    domain: str,
    target: str,
    overrides: ContractOverrides | None = None,
) -> TableContractSpec:
    """Return a table contract spec, raising if it cannot be resolved.

    Returns
    -------
    TableContractSpec
        Resolved contract spec.
    """
    registry = get_contract_registry()
    return registry.require_contract(
        table_key=table_key,
        domain=domain,
        target=target,
        overrides=overrides,
    )


def require_contract_for_target(
    *,
    table_key: str,
    target_name: str,
    overrides: ContractOverrides | None = None,
) -> TableContractSpec:
    """Return a table contract spec for a target.

    Returns
    -------
    TableContractSpec
        Resolved contract spec.
    """
    domain = _domain_from_table_key(table_key)
    return require_contract(
        table_key=table_key,
        domain=domain,
        target=target_name,
        overrides=overrides,
    )


def contract_for_table(
    *,
    table_key: str,
    target_name: str,
    input_name: str,
    **overrides: Unpack[ContractForTableOverrides],
) -> TableContractSpec:
    """Return a contract spec for a target table with override inputs.

    Returns
    -------
    TableContractSpec
        Resolved contract spec.
    """
    resolved_overrides = ContractOverrides(
        input_name=input_name,
        **overrides,
    )
    return require_contract_for_target(
        table_key=table_key,
        target_name=target_name,
        overrides=resolved_overrides,
    )


def contracts_for_target(
    *,
    target_name: str,
    specs: Sequence[ContractForTableInput],
) -> tuple[TableContractSpec, ...]:
    """Return contract specs for a target from per-table inputs.

    Returns
    -------
    tuple[TableContractSpec, ...]
        Resolved contract specs.
    """
    return tuple(
        contract_for_table(
            table_key=spec.table_key,
            target_name=target_name,
            input_name=spec.input_name,
            **_overrides_from_spec(spec),
        )
        for spec in specs
    )


def _overrides_from_spec(spec: ContractForTableInput) -> ContractForTableOverrides:
    overrides: ContractForTableOverrides = {}
    if spec.ops_module is not UNSET:
        overrides["ops_module"] = cast("ModuleType | None", spec.ops_module)
    if spec.columns_to_pass is not UNSET:
        overrides["columns_to_pass"] = cast("Sequence[str]", spec.columns_to_pass)
    if spec.required_cols is not UNSET:
        overrides["required_cols"] = cast("Sequence[str]", spec.required_cols)
    if spec.clip_column is not UNSET:
        overrides["clip_column"] = cast("str | None", spec.clip_column)
    if spec.policy is not UNSET:
        overrides["policy"] = cast("ContractPolicy", spec.policy)
    return overrides


__all__ = [
    "ContractForTableInput",
    "ContractRegistry",
    "ContractResolver",
    "SchemaBackedContractRegistry",
    "contract_descriptor_for_table_key",
    "contract_descriptor_for_table_schema",
    "contract_for_table",
    "contracts_for_target",
    "get_contract",
    "get_contract_for_target",
    "get_contract_registry",
    "require_contract",
    "require_contract_for_target",
    "set_contract_registry",
]
