"""Composable analytics profile recipes for functions, files, and modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.analytics.profiles.files import build_file_profile as _build_file_profile
from codeintel.analytics.profiles.functions import (
    SLOW_TEST_THRESHOLD_MS,
    build_function_profile_recipe,
)
from codeintel.analytics.profiles.modules import build_module_profile as _build_module_profile
from codeintel.analytics.profiles.utils import seed_catalog_modules
from codeintel.graphs.catalog import (
    FunctionCatalogService,
)

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.catalog import (
        FunctionCatalogProvider,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def build_function_profile(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    module_map: dict[str, str] | None = None,
) -> None:
    """Populate analytics.function_profile for a snapshot.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.
    catalog_provider
        Optional function catalog provider.
    module_map
        Optional module map override.
    """
    effective_catalog = catalog_provider or FunctionCatalogService.from_db(
        gateway,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    module_table = seed_catalog_modules(
        gateway,
        effective_catalog,
        snapshot.repo,
        snapshot.commit,
        module_map_override=module_map,
    )
    count = build_function_profile_recipe(gateway, snapshot, module_table=module_table)
    log.info("function_profile populated: %s rows for %s@%s", count, snapshot.repo, snapshot.commit)


def build_file_profile(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    module_map: dict[str, str] | None = None,
) -> None:
    """Populate analytics.file_profile by aggregating function_profile.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.
    catalog_provider
        Optional function catalog provider.
    module_map
        Optional module map override.
    """
    module_table = seed_catalog_modules(
        gateway,
        catalog_provider,
        snapshot.repo,
        snapshot.commit,
        module_map_override=module_map,
    )
    count = _build_file_profile(gateway, snapshot, module_table=module_table)
    log.info("file_profile populated: %s rows for %s@%s", count, snapshot.repo, snapshot.commit)


def build_module_profile(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    module_map: dict[str, str] | None = None,
) -> None:
    """Populate analytics.module_profile by aggregating file/function profiles.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.
    catalog_provider
        Optional function catalog provider.
    module_map
        Optional module map override.
    """
    module_table = seed_catalog_modules(
        gateway,
        catalog_provider,
        snapshot.repo,
        snapshot.commit,
        module_map_override=module_map,
    )
    count = _build_module_profile(gateway, snapshot, module_table=module_table)
    log.info("module_profile populated: %s rows for %s@%s", count, snapshot.repo, snapshot.commit)


__all__ = [
    "SLOW_TEST_THRESHOLD_MS",
    "build_file_profile",
    "build_function_profile",
    "build_module_profile",
]
