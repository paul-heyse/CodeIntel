"""Shared environment builder for tests.

Provides a single entry point for constructing gateways and ``TestContext``
instances with production-parity defaults (schema, views, macros). Exports
canonical test defaults for repo/commit/run identifiers.
"""

from __future__ import annotations

from pathlib import Path

from codeintel.storage.gateway import StorageGateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
from tests._helpers.context import (
    TestContext,
    create_test_context,
)
from tests._helpers.context import (
    build_test_gateway as _build_test_gateway,
)
from tests._helpers.env_options import EnvOptions, GatewayOptions


def build_test_gateway(
    options: GatewayOptions | None = None,
) -> StorageGateway:
    """Create a StorageGateway with schema/views/macros ensured.

    Parameters
    ----------
    options
        Gateway configuration bundle.

    Returns
    -------
    StorageGateway
        Gateway ready for test use with macros ensured.
    """
    return _build_test_gateway(options)


def create_test_env(
    tmp_path: Path,
    *,
    options: EnvOptions | None = None,
    gateway_options: GatewayOptions | None = None,
) -> TestContext:
    """Build a TestContext with consistent defaults via GatewayFactory.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    options
        Environment options bundle overriding repo, commit, and filesystem paths.
    gateway_options
        Optional gateway options bundle for schema/view/validation overrides.

    Returns
    -------
    TestContext
        Configured test context with gateway, snapshot, and build paths.
    """
    return create_test_context(tmp_path, options=options, gateway_options=gateway_options)


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "build_test_gateway",
    "create_test_env",
]
