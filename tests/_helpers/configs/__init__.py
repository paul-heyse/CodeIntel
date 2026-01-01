"""Configuration dataclasses for test environments.

This module provides configuration and environment dataclasses that define
the structure of test environments without containing orchestration logic.
"""

from __future__ import annotations

from tests._helpers.configs.coverage_config import CoverageEdgeEnv, CoverageSeedConfig
from tests._helpers.configs.graph_config import GraphEngineSeed, SpanSnapshot, SpanTestEnv
from tests._helpers.configs.provisioning_config import (
    CallgraphFixtureOptions,
    GatewayOptions,
    GraphMetricsGatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    ProvisioningSetup,
    ProvisionOptions,
    RepoContext,
    provisioning_gateway_options,
)
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT, SnapshotVariant

__all__ = [
    "DEFAULT_VARIANT",
    "CallgraphFixtureOptions",
    "CoverageEdgeEnv",
    "CoverageSeedConfig",
    "GatewayOptions",
    "GraphEngineSeed",
    "GraphMetricsGatewayOptions",
    "ProvisionOptions",
    "ProvisionedGateway",
    "ProvisioningConfig",
    "ProvisioningSetup",
    "RepoContext",
    "SnapshotVariant",
    "SpanSnapshot",
    "SpanTestEnv",
    "provisioning_gateway_options",
]
