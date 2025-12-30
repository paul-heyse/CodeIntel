"""Validation policy helpers for materialization-time checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.core.config.view import SettingsView
from codeintel.core.schemas.primitives import TableSchema

ValidationScope = Literal["contract", "internal"]

_VALIDATION_PROFILE_TAG = "ci.validation_profile"


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Resolved validation policy for a materialized output."""

    table_key: str
    output_role: str
    scope: ValidationScope
    profile: str | None
    enabled: bool
    disabled_reason: str | None = None

    @property
    def run_contract_checks(self) -> bool:
        """Return True when contract checks should run.

        Returns
        -------
        bool
            True when contract validation is enabled.
        """
        return self.enabled and self.scope == "contract"

    @property
    def run_internal_checks(self) -> bool:
        """Return True when internal checks should run.

        Returns
        -------
        bool
            True when internal validation is enabled.
        """
        return self.enabled


def resolve_validation_policy(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
    table_key: str,
    output_role: str | None,
    declared_schema: TableSchema | None,
) -> ValidationPolicy:
    """Resolve validation policy from DAG tags and runtime settings.

    Returns
    -------
    ValidationPolicy
        Resolved policy with enablement and scope decisions.
    """
    output = catalog.table_outputs.get(table_key)
    resolved_role = output.role if output is not None else (output_role or "contract")
    profile_tag = _validation_profile_tag(output.tags if output is not None else None)
    profile = SettingsView.resolve_validation_profile(
        default_profile=profile_tag,
        config_mode=env.validation_mode.value,
    )
    if not env.validate_outputs:
        profile = None
    enabled = env.validate_outputs and profile is not None
    scope: ValidationScope = (
        "contract" if resolved_role == "contract" and declared_schema is not None else "internal"
    )
    disabled_reason = _disabled_reason(
        env=env,
        scope=scope,
        role=resolved_role,
        declared_schema=declared_schema,
        profile=profile,
    )
    return ValidationPolicy(
        table_key=table_key,
        output_role=resolved_role,
        scope=scope,
        profile=profile,
        enabled=enabled,
        disabled_reason=disabled_reason,
    )


def _validation_profile_tag(tags: object | None) -> str | None:
    if not isinstance(tags, dict):
        return None
    raw = tags.get(_VALIDATION_PROFILE_TAG)
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    if normalized in {"strict", "lenient"}:
        return normalized
    return None


def _disabled_reason(
    *,
    env: BuildEnv,
    scope: ValidationScope,
    role: str,
    declared_schema: TableSchema | None,
    profile: str | None,
) -> str | None:
    if not env.validate_outputs:
        return "validation_disabled"
    if profile is None:
        return "validation_mode_off"
    if role != "contract":
        return "output_role_internal"
    if scope != "contract" or declared_schema is None:
        return "missing_declared_schema"
    return None


__all__ = ["ValidationPolicy", "ValidationScope", "resolve_validation_policy"]
