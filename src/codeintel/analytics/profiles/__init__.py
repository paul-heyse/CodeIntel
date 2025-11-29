"""Composable analytics profile recipes for functions, files, and modules."""

from __future__ import annotations

import logging

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.profiles.files import build_file_profile as _build_file_profile
from codeintel.analytics.profiles.functions import (
    SLOW_TEST_THRESHOLD_MS,
    build_function_profile_recipe,
)
from codeintel.analytics.profiles.modules import build_module_profile as _build_module_profile
from codeintel.analytics.profiles.utils import seed_catalog_modules
from codeintel.config import ProfilesAnalyticsStepConfig
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def build_function_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    context: AnalyticsContext | None = None,
) -> None:
    """Populate analytics.function_profile for a snapshot."""
    effective_catalog = context.catalog if context is not None else catalog_provider
    module_map_override = context.module_map if context is not None else None
    if effective_catalog is None:
        effective_catalog = FunctionCatalogService.from_db(
            gateway,
            repo=cfg.repo,
            commit=cfg.commit,
        )
    module_table = seed_catalog_modules(
        gateway.con,
        effective_catalog,
        cfg.repo,
        cfg.commit,
        module_map_override=module_map_override,
    )
    count = build_function_profile_recipe(gateway, cfg, module_table=module_table)
    log.info("function_profile populated: %s rows for %s@%s", count, cfg.repo, cfg.commit)


def build_file_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    context: AnalyticsContext | None = None,
) -> None:
    """Populate analytics.file_profile by aggregating function_profile."""
    effective_catalog = context.catalog if context is not None else catalog_provider
    module_map_override = context.module_map if context is not None else None
    module_table = seed_catalog_modules(
        gateway.con,
        effective_catalog,
        cfg.repo,
        cfg.commit,
        module_map_override=module_map_override,
    )
    count = _build_file_profile(gateway, cfg, module_table=module_table)
    log.info("file_profile populated: %s rows for %s@%s", count, cfg.repo, cfg.commit)


def build_module_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    context: AnalyticsContext | None = None,
) -> None:
    """Populate analytics.module_profile by aggregating file/function profiles."""
    effective_catalog = context.catalog if context is not None else catalog_provider
    module_map_override = context.module_map if context is not None else None
    module_table = seed_catalog_modules(
        gateway.con,
        effective_catalog,
        cfg.repo,
        cfg.commit,
        module_map_override=module_map_override,
    )
    count = _build_module_profile(gateway, cfg, module_table=module_table)
    log.info("module_profile populated: %s rows for %s@%s", count, cfg.repo, cfg.commit)


__all__ = [
    "SLOW_TEST_THRESHOLD_MS",
    "build_file_profile",
    "build_function_profile",
    "build_module_profile",
]
