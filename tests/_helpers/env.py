"""Shared environment builder for tests.

Provides a single entry point for constructing gateways and ``TestContext``
instances with production-parity defaults (schema, views, macros). Exports
canonical test defaults for repo/commit/run identifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

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

if TYPE_CHECKING:
    from typing import Unpack


class _LegacyGatewayKwargs(TypedDict, total=False):
    file_backed: bool
    db_path: Path
    repo: str
    commit: str
    apply_schema: bool
    ensure_views: bool
    validate_schema: bool


class _LegacyEnvKwargs(TypedDict, total=False):
    repo: str
    commit: str
    file_backed: bool
    repo_root: Path
    build_dir: Path
    db_path: Path


def build_test_gateway(
    options: GatewayOptions | None = None,
    **legacy: Unpack[_LegacyGatewayKwargs],
) -> StorageGateway:
    """Create a StorageGateway with schema/views/macros ensured.

    Parameters
    ----------
    options
        Gateway configuration bundle.
    legacy
        Backward-compatible keyword overrides mapped onto GatewayOptions.

    Returns
    -------
    StorageGateway
        Gateway ready for test use with macros ensured.
    """
    merged = options or _merge_gateway_options(legacy)
    return _build_test_gateway(merged)


def create_test_env(
    tmp_path: Path,
    *,
    options: EnvOptions | None = None,
    gateway_options: GatewayOptions | None = None,
    **legacy: Unpack[_LegacyEnvKwargs],
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
    legacy
        Backward-compatible keyword overrides mapped onto EnvOptions.

    Returns
    -------
    TestContext
        Configured test context with gateway, snapshot, and build paths.
    """
    env_options = options or _merge_env_options(legacy)
    return create_test_context(tmp_path, options=env_options, gateway_options=gateway_options)


def _merge_gateway_options(legacy: _LegacyGatewayKwargs) -> GatewayOptions:
    """Convert legacy keyword arguments into a GatewayOptions instance.

    Returns
    -------
    GatewayOptions
        Options populated from legacy keyword arguments.
    """
    return GatewayOptions(
        file_backed=legacy.get("file_backed", False),
        db_path=legacy.get("db_path"),
        repo=legacy.get("repo"),
        commit=legacy.get("commit"),
        apply_schema=legacy.get("apply_schema", True),
        ensure_views=legacy.get("ensure_views", True),
        validate_schema=legacy.get("validate_schema", True),
    )


def _merge_env_options(legacy: _LegacyEnvKwargs) -> EnvOptions:
    """Convert legacy keyword arguments into an EnvOptions instance.

    Returns
    -------
    EnvOptions
        Options populated from legacy keyword arguments.
    """
    return EnvOptions(
        repo=legacy.get("repo", DEFAULT_REPO),
        commit=legacy.get("commit", DEFAULT_COMMIT),
        file_backed=legacy.get("file_backed", False),
        repo_root=legacy.get("repo_root"),
        build_dir=legacy.get("build_dir"),
        db_path=legacy.get("db_path"),
    )


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "build_test_gateway",
    "create_test_env",
]
