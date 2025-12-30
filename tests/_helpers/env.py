"""Test environment and gateway helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

from tests._helpers.configs.provisioning_config import ProvisioningConfig
from tests._helpers.context import (
    TestContext,
    create_test_context,
)
from tests._helpers.context import (
    build_test_gateway as _build_test_gateway,
)
from tests._helpers.env_options import EnvOptions, GatewayOptions
from tests._helpers.fixtures.snapshots import SnapshotVariant
from tests._helpers.orchestration.provisioning import (
    provision_gateway_with_repo,
    provision_hamilton_repo,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


@contextmanager
def temporary_env(
    overrides: Mapping[str, str | None] | None = None,
    **kwargs: str | None,
) -> Iterator[None]:
    """Temporarily set environment variables within a context.

    Parameters
    ----------
    overrides
        Optional mapping of environment variable names to values.
        Use None to unset a variable.
    **kwargs
        Keyword overrides merged with the mapping.
    """
    combined = dict(overrides or {})
    combined.update(kwargs)
    prior = {key: os.environ.get(key) for key in combined}
    try:
        for key, value in combined.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def unset_env(key: str) -> Iterator[None]:
    """Temporarily unset an environment variable for the duration of the context.

    Parameters
    ----------
    key
        Environment variable name.
    """
    with temporary_env({key: None}):
        yield


def build_test_gateway(options: GatewayOptions | None = None) -> StorageGateway:
    """Create a StorageGateway with schemas and views ensured.

    Returns
    -------
    StorageGateway
        Initialized storage gateway.
    """
    return _build_test_gateway(options)


def create_test_env(
    tmp_path: Path,
    options: EnvOptions | None = None,
    *,
    gateway_options: GatewayOptions | None = None,
    snapshot_variant: SnapshotVariant | None = None,
) -> TestContext:
    """Create a TestContext configured for test runs.

    Returns
    -------
    TestContext
        Provisioned test context wrapper.
    """
    return create_test_context(
        tmp_path,
        options,
        gateway_options=gateway_options,
        snapshot_variant=snapshot_variant,
    )


def create_provisioned_test_env(
    repo_root: Path,
    config: ProvisioningConfig | None = None,
) -> TestContext:
    """Provision a seeded gateway and return a TestContext wrapper.

    Returns
    -------
    TestContext
        Provisioned test context wrapper.
    """
    cfg = config or ProvisioningConfig()
    if cfg.run_ingestion:
        provisioned = provision_hamilton_repo(
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
    "build_test_gateway",
    "create_provisioned_test_env",
    "create_test_env",
    "temporary_env",
    "unset_env",
]
