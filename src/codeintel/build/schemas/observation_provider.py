"""Helpers for resolving schema observation providers in build workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.schemas.bundle_observations import BundleSchemaObservationProvider
from codeintel.core.schemas.resolution import SchemaObservationProvider

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv


def observation_provider_for_env(env: BuildEnv) -> SchemaObservationProvider | None:
    """Return the best available schema observation provider for a build env.

    Returns
    -------
    SchemaObservationProvider | None
        Provider instance when metadata bundle is available, otherwise None.
    """
    if env.metadata_bundle is not None:
        return BundleSchemaObservationProvider(env.metadata_bundle.bundle_root)
    return None


__all__ = ["observation_provider_for_env"]
