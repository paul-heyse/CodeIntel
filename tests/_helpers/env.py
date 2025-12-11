"""Shared environment builder for tests.

Provides a single entry point for constructing gateways and ``TestContext``
instances with production-parity defaults (schema, views). Exports canonical
test defaults for repo/commit/run identifiers.
"""

from __future__ import annotations

from pathlib import Path

from codeintel.storage.gateway import StorageGateway
from tests._helpers.configs.provisioning_config import ProvisioningConfig
from tests._helpers.context import (
    TestContext,
    create_test_context,
)
from tests._helpers.context import (
    build_test_gateway as _build_test_gateway,
)
from tests._helpers.defaults import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
from tests._helpers.env_options import EnvOptions, GatewayOptions
from tests._helpers.orchestration.provisioning import (
    provision_gateway_with_repo,
    provision_ingested_repo,
)


def build_test_gateway(
    options: GatewayOptions | None = None,
) -> StorageGateway:
    """Create a StorageGateway with schema/views ensured.

    Parameters
    ----------
    options
        Gateway configuration bundle.

    Returns
    -------
    StorageGateway
        Gateway ready for test use with schemas/views applied.
    """
    return _build_test_gateway(options)


def create_test_env(
    tmp_path: Path,
    *,
    options: EnvOptions | None = None,
    gateway_options: GatewayOptions | None = None,
) -> TestContext:
    """Build a TestContext with consistent defaults via GatewayFactory.

    Returns
    -------
    TestContext
        Constructed context with schemas/views ensured.
    """
    return create_test_context(tmp_path, options=options, gateway_options=gateway_options)


def create_provisioned_test_env(
    repo_root: Path,
    config: ProvisioningConfig | None = None,
) -> TestContext:
    """Build a TestContext using provisioning flows (ingested or schema-only).

    Parameters
    ----------
    repo_root
        Root path where the repo/build artifacts should live.
    config
        Provisioning configuration; defaults mirror ProvisioningConfig.

    Returns
    -------
    TestContext
        Provisioned context with gateway, snapshot, and build paths.
    """
    cfg = config or ProvisioningConfig()
    if cfg.run_ingestion:
        provisioned = provision_ingested_repo(
            repo_root,
            repo=cfg.repo,
            commit=cfg.commit,
            options=cfg.provision_options,
        )
    else:
        provisioned = provision_gateway_with_repo(
            repo_root,
            repo=cfg.repo,
            commit=cfg.commit,
            options=cfg.gateway_options,
        )
    return TestContext.from_provisioned(provisioned)


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "build_test_gateway",
    "create_provisioned_test_env",
    "create_test_env",
]
