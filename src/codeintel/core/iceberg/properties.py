"""Helpers for Iceberg table property configuration."""

from __future__ import annotations

from pyiceberg.table import TableProperties

from codeintel.core.config.settings import IcebergSettings
from codeintel.core.serialization.stable import stable_stringify


def iceberg_location_properties(settings: IcebergSettings) -> dict[str, str]:
    """Return table property overrides for Iceberg location providers.

    Returns
    -------
    dict[str, str]
        Property overrides for Iceberg table configuration.
    """
    properties: dict[str, str] = {}
    if settings.location_provider_impl:
        properties[TableProperties.WRITE_PY_LOCATION_PROVIDER_IMPL] = (
            settings.location_provider_impl
        )
    if settings.write_data_path:
        properties[TableProperties.WRITE_DATA_PATH] = settings.write_data_path
    if settings.write_metadata_path:
        properties[TableProperties.WRITE_METADATA_PATH] = settings.write_metadata_path
    if settings.object_store_partitioned_paths is not None:
        properties[TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS] = stable_stringify(
            settings.object_store_partitioned_paths
        )
    return properties


__all__ = ["iceberg_location_properties"]
