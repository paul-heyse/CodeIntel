"""Policy registry for contract resolution defaults."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import cast

from codeintel.build.config import BuildConfig
from codeintel.build.contracts.types import UNSET, ContractOverrides, ContractPolicy
from codeintel.core.schemas.arrow_gen import EXTRAS_POLICIES, ExtrasPolicy
from codeintel.core.validation.profiles import (
    ValidationProfile,
    normalize_validation_profile,
)

_DEFAULT_POLICY = ContractPolicy()
_CONTRACT_KEYS = frozenset(
    {
        "default_profile",
        "policy_profiles",
        "policy_tables",
        "policy_targets",
    }
)


@dataclass(slots=True)
class ContractPolicyRegistry:
    """Registry for resolving contract policy profiles."""

    profiles: dict[str, ContractPolicy] = field(default_factory=dict)
    table_profiles: dict[str, str] = field(default_factory=dict)
    target_profiles: dict[str, str] = field(default_factory=dict)
    default_profile: str | None = None

    def register_profile(
        self,
        *,
        name: str,
        extras_policy: ExtrasPolicy | None = None,
        validation_profile: ValidationProfile | None = None,
        coerce_types: bool | None = None,
        allow_nulls: bool | None = None,
    ) -> None:
        """Register a named policy profile.

        Raises
        ------
        ValueError
            Raised when the profile name or extras policy is invalid.
        """
        if not name:
            msg = "Policy profile name must be non-empty"
            raise ValueError(msg)
        if name in self.profiles:
            msg = f"Policy profile already registered: {name!r}"
            raise ValueError(msg)
        if extras_policy is not None and extras_policy not in EXTRAS_POLICIES:
            msg = f"Invalid extras_policy: {extras_policy!r}"
            raise ValueError(msg)
        policy = ContractPolicy(
            extras_policy=extras_policy,
            validation_profile=validation_profile,
            coerce_types=_DEFAULT_POLICY.coerce_types if coerce_types is None else coerce_types,
            allow_nulls=_DEFAULT_POLICY.allow_nulls if allow_nulls is None else allow_nulls,
        )
        self.profiles[name] = policy

    def attach_table_profile(self, table_key: str, profile_name: str) -> None:
        """Attach a profile to a specific table key."""
        self.require_profile(profile_name)
        self.table_profiles[table_key] = profile_name

    def attach_target_profile(self, target_name: str, profile_name: str) -> None:
        """Attach a profile to a target name."""
        self.require_profile(profile_name)
        self.target_profiles[target_name] = profile_name

    def resolve_policy(self, *, table_key: str, target_name: str | None) -> ContractPolicy | None:
        """Resolve a policy profile for a table/target pair.

        Returns
        -------
        ContractPolicy | None
            Resolved policy or None if no profile matches.
        """
        profile_name = self.table_profiles.get(table_key)
        if profile_name is None and target_name is not None:
            profile_name = self.target_profiles.get(target_name)
        if profile_name is None:
            profile_name = self.default_profile
        if profile_name is None:
            return None
        return self.require_profile(profile_name)

    def require_profile(self, name: str) -> ContractPolicy:
        """Return a policy profile by name.

        Returns
        -------
        ContractPolicy
            Resolved policy profile.

        Raises
        ------
        KeyError
            Raised when the profile name is unknown.
        """
        policy = self.profiles.get(name)
        if policy is None:
            msg = f"Unknown contract policy profile: {name!r}"
            raise KeyError(msg)
        return policy


def policy_registry_from_config(config: BuildConfig) -> ContractPolicyRegistry:
    """Build a ContractPolicyRegistry from build config.

    Returns
    -------
    ContractPolicyRegistry
        Configured policy registry derived from config.

    Raises
    ------
    TypeError
        Raised when the contracts section is not a mapping.
    """
    raw = config.get("contracts")
    if raw is None:
        return ContractPolicyRegistry()
    if not isinstance(raw, Mapping):
        msg = f"contracts section must be a mapping in {_config_label(config)}"
        raise TypeError(msg)
    return _policy_registry_from_mapping(raw, label=_config_label(config))


def configure_contract_policy_registry(*, config: BuildConfig) -> ContractPolicyRegistry:
    """Configure and return the global contract policy registry.

    Returns
    -------
    ContractPolicyRegistry
        Configured contract policy registry.
    """
    registry = policy_registry_from_config(config)
    set_contract_policy_registry(registry=registry)
    return registry


def apply_policy_overrides(
    *,
    table_key: str,
    target_name: str,
    overrides: ContractOverrides,
    registry: ContractPolicyRegistry | None = None,
) -> ContractOverrides:
    """Apply registry policy defaults to contract overrides.

    Returns
    -------
    ContractOverrides
        Updated overrides with policy defaults applied when available.
    """
    if overrides.policy is not UNSET:
        return overrides
    resolved_registry = registry or get_contract_policy_registry()
    policy = resolved_registry.resolve_policy(table_key=table_key, target_name=target_name)
    if policy is None:
        return overrides
    return replace(overrides, policy=policy)


def set_contract_policy_registry(*, registry: ContractPolicyRegistry | None) -> None:
    """Set the global contract policy registry."""
    _POLICY_REGISTRY_STATE["registry"] = registry


def get_contract_policy_registry() -> ContractPolicyRegistry:
    """Return the configured contract policy registry.

    Returns
    -------
    ContractPolicyRegistry
        Configured registry or an empty default when unconfigured.
    """
    registry = _POLICY_REGISTRY_STATE["registry"]
    if registry is None:
        registry = ContractPolicyRegistry()
        _POLICY_REGISTRY_STATE["registry"] = registry
    return registry


def _config_label(config: BuildConfig) -> str:
    if config.config_path is None:
        return "build config"
    return str(config.config_path)


def _policy_registry_from_mapping(
    raw: Mapping[str, object],
    *,
    label: str,
) -> ContractPolicyRegistry:
    unknown = sorted(set(raw) - _CONTRACT_KEYS)
    if unknown:
        msg = f"Unknown contracts config keys in {label}: {', '.join(unknown)}"
        raise ValueError(msg)

    registry = ContractPolicyRegistry()
    profiles_raw = raw.get("policy_profiles")
    if profiles_raw is not None:
        _load_profiles(registry, profiles_raw, label=label)

    default_profile = _optional_str(raw.get("default_profile"))
    if default_profile is not None:
        registry.default_profile = default_profile
        registry.require_profile(default_profile)

    tables_raw = raw.get("policy_tables")
    if tables_raw is not None:
        _load_profile_map(
            registry.attach_table_profile,
            tables_raw,
            label=label,
            entry_label="policy_tables",
        )

    targets_raw = raw.get("policy_targets")
    if targets_raw is not None:
        _load_profile_map(
            registry.attach_target_profile,
            targets_raw,
            label=label,
            entry_label="policy_targets",
        )

    return registry


def _load_profiles(
    registry: ContractPolicyRegistry,
    profiles_raw: object,
    *,
    label: str,
) -> None:
    if not isinstance(profiles_raw, Mapping):
        msg = f"contracts.policy_profiles must be a mapping in {label}"
        raise TypeError(msg)
    for name, raw in profiles_raw.items():
        if not isinstance(name, str) or not name:
            msg = f"contracts.policy_profiles keys must be strings in {label}"
            raise TypeError(msg)
        policy = _parse_profile(raw, label=label, profile_name=name)
        registry.register_profile(
            name=name,
            extras_policy=policy.extras_policy,
            validation_profile=policy.validation_profile,
            coerce_types=policy.coerce_types,
            allow_nulls=policy.allow_nulls,
        )


def _parse_profile(
    raw: object,
    *,
    label: str,
    profile_name: str,
) -> ContractPolicy:
    if not isinstance(raw, Mapping):
        msg = f"contracts.policy_profiles[{profile_name!r}] must be a mapping in {label}"
        raise TypeError(msg)
    extras_policy = _parse_extras_policy(raw.get("extras_policy"), label=label)
    validation_profile = _parse_validation_profile(raw.get("validation_profile"), label=label)
    coerce_types = _parse_optional_bool(raw.get("coerce_types"), label=label)
    allow_nulls = _parse_optional_bool(raw.get("allow_nulls"), label=label)
    return ContractPolicy(
        extras_policy=extras_policy,
        validation_profile=validation_profile,
        coerce_types=_DEFAULT_POLICY.coerce_types if coerce_types is None else coerce_types,
        allow_nulls=_DEFAULT_POLICY.allow_nulls if allow_nulls is None else allow_nulls,
    )


def _load_profile_map(
    attach: Callable[[str, str], None],
    raw: object,
    *,
    label: str,
    entry_label: str,
) -> None:
    if not isinstance(raw, Mapping):
        msg = f"contracts.{entry_label} must be a mapping in {label}"
        raise TypeError(msg)
    for key, profile_name in raw.items():
        if not isinstance(key, str) or not key:
            msg = f"contracts.{entry_label} keys must be strings in {label}"
            raise TypeError(msg)
        if not isinstance(profile_name, str) or not profile_name:
            msg = f"contracts.{entry_label} values must be strings in {label}"
            raise TypeError(msg)
        attach(key, profile_name)


def _parse_extras_policy(value: object, *, label: str) -> ExtrasPolicy | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"contracts.policy_profiles extras_policy must be a string in {label}"
        raise TypeError(msg)
    normalized = value.strip().lower()
    if normalized in EXTRAS_POLICIES:
        return cast("ExtrasPolicy", normalized)
    msg = f"Invalid extras_policy {value!r} in {label}"
    raise ValueError(msg)


def _parse_validation_profile(
    value: object,
    *,
    label: str,
) -> ValidationProfile | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"contracts.policy_profiles validation_profile must be a string in {label}"
        raise TypeError(msg)
    return normalize_validation_profile(value)


def _parse_optional_bool(value: object, *, label: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    msg = f"contracts.policy_profiles boolean fields must be bools in {label}"
    raise TypeError(msg)


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


_POLICY_REGISTRY_STATE: dict[str, ContractPolicyRegistry | None] = {"registry": None}


__all__ = [
    "ContractPolicyRegistry",
    "apply_policy_overrides",
    "configure_contract_policy_registry",
    "get_contract_policy_registry",
    "policy_registry_from_config",
    "set_contract_policy_registry",
]
