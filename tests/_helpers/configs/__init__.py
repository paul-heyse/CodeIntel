"""Configuration dataclasses for test environments.

This module provides configuration and environment dataclasses that define
the structure of test environments without containing orchestration logic.
"""

from __future__ import annotations

from tests._helpers.configs.coverage_config import CoverageEdgeEnv, CoverageSeedConfig
from tests._helpers.configs.graph_config import GraphEngineSeed, SpanSnapshot, SpanTestEnv
from tests._helpers.configs.pipeline_config import PipelineEnv
from tests._helpers.configs.provisioning_config import (
    DEFAULT_COMMIT,
    DEFAULT_REPO,
    CallgraphFixtureOptions,
    GatewayOptions,
    GraphMetricsGatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    ProvisioningSetup,
    ProvisionOptions,
    RepoContext,
)

__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "CallgraphFixtureOptions",
    "CoverageEdgeEnv",
    "CoverageSeedConfig",
    "GatewayOptions",
    "GraphEngineSeed",
    "GraphMetricsGatewayOptions",
    "PipelineEnv",
    "ProvisionOptions",
    "ProvisionedGateway",
    "ProvisioningConfig",
    "ProvisioningSetup",
    "RepoContext",
    "SpanSnapshot",
    "SpanTestEnv",
]
