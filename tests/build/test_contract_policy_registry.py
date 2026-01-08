"""Contract policy registry behavior tests."""

from __future__ import annotations

from typing import cast

from codeintel.build.config import BuildConfig
from codeintel.build.contracts.policy_registry import (
    ContractPolicyRegistry,
    apply_policy_overrides,
    policy_registry_from_config,
)
from codeintel.build.contracts.types import UNSET, ContractOverrides, ContractPolicy


def test_policy_registry_resolves_table_over_target() -> None:
    """Prefer table-level profiles over target-level profiles."""
    registry = ContractPolicyRegistry()
    registry.register_profile(name="strict", extras_policy="reject", coerce_types=False)
    registry.register_profile(name="lenient", extras_policy="retain")
    registry.attach_target_profile("graph_metrics", "lenient")
    registry.attach_table_profile("analytics.graph_metrics_functions", "strict")

    resolved = registry.resolve_policy(
        table_key="analytics.graph_metrics_functions",
        target_name="graph_metrics",
    )

    assert resolved is not None
    assert resolved.extras_policy == "reject"
    assert resolved.coerce_types is False


def test_apply_policy_overrides_sets_default_profile() -> None:
    """Apply default profile policy when overrides are unset."""
    registry = ContractPolicyRegistry()
    registry.register_profile(name="strict", extras_policy="reject")
    registry.default_profile = "strict"
    overrides = ContractOverrides()

    resolved = apply_policy_overrides(
        table_key="analytics.function_types",
        target_name="contract_ref_test",
        overrides=overrides,
        registry=registry,
    )

    assert resolved.policy is not UNSET
    policy = cast("ContractPolicy", resolved.policy)
    assert policy.extras_policy == "reject"


def test_policy_registry_from_config_reads_profiles() -> None:
    """Load policy profiles and mappings from BuildConfig."""
    config = BuildConfig.from_dict(
        {
            "contracts": {
                "policy_profiles": {
                    "strict": {"extras_policy": "reject", "coerce_types": False},
                },
                "policy_tables": {"analytics.function_types": "strict"},
            }
        }
    )

    registry = policy_registry_from_config(config)
    resolved = registry.resolve_policy(
        table_key="analytics.function_types",
        target_name="function_types",
    )

    assert resolved is not None
    assert resolved.extras_policy == "reject"
    assert resolved.coerce_types is False
