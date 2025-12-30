"""PyIceberg catalog loading and table identifier helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pyiceberg.catalog import Catalog, load_catalog

from codeintel.core.config.settings import IcebergSettings
from codeintel.storage.helpers.table_key import parse_table_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from pyiceberg.table import Table


@contextmanager
def _pyiceberg_home(config_path: Path | None) -> Iterator[Mapping[str, str]]:
    if config_path is None:
        yield os.environ
        return
    home_path = config_path
    if home_path.is_file():
        home_path = home_path.parent
    prior = os.environ.get("PYICEBERG_HOME")
    os.environ["PYICEBERG_HOME"] = str(home_path)
    try:
        yield os.environ
    finally:
        if prior is None:
            os.environ.pop("PYICEBERG_HOME", None)
        else:
            os.environ["PYICEBERG_HOME"] = prior


@dataclass(frozen=True, slots=True)
class IcebergCatalogProvider:
    """Resolve Iceberg catalog connections and table identifiers."""

    settings: IcebergSettings

    def load(self) -> Catalog:
        """Load the configured Iceberg catalog.

        Returns
        -------
        pyiceberg.catalog.Catalog
            Loaded catalog instance.
        """
        properties = self._catalog_properties()
        catalog_name = self.settings.catalog_name or None
        with _pyiceberg_home(self.settings.config_path):
            return load_catalog(catalog_name, **properties)

    def load_table(self, table_key: str) -> Table:
        """Load an Iceberg table for the provided table key.

        Returns
        -------
        pyiceberg.table.Table
            Loaded Iceberg table instance.
        """
        catalog = self.load()
        identifier = self.resolve_identifier(table_key)
        return catalog.load_table(identifier)

    def table_exists(self, table_key: str) -> bool:
        """Return True when the catalog contains the table key.

        Returns
        -------
        bool
            True when the table exists in the catalog.
        """
        catalog = self.load()
        identifier = self.resolve_identifier(table_key)
        return bool(catalog.table_exists(identifier))

    @staticmethod
    def resolve_identifier(table_key: str) -> tuple[str, ...]:
        """Resolve a table key into a catalog identifier tuple.

        Returns
        -------
        tuple[str, ...]
            Catalog identifier derived from the table key.
        """
        parsed = parse_table_key(table_key)
        return (parsed.schema, parsed.name)

    def _catalog_properties(self) -> dict[str, str]:
        properties: dict[str, str] = {}
        if self.settings.catalog_type is not None:
            if self.settings.catalog_type != "sql":
                msg = "Only the Iceberg SQL catalog is supported."
                raise ValueError(msg)
            properties["type"] = self.settings.catalog_type
        if self.settings.catalog_uri is not None:
            properties["uri"] = self.settings.catalog_uri
        if self.settings.catalog_warehouse is not None:
            properties["warehouse"] = self.settings.catalog_warehouse
        if self.settings.io_impl is not None:
            properties["io.impl"] = self.settings.io_impl
        properties.update(dict(self.settings.catalog_properties))
        properties.update(dict(self.settings.io_options))
        return properties


__all__ = ["IcebergCatalogProvider"]
