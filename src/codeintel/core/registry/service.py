"""Canonical registry service for datasets, targets, and semantic views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.catalogs.canonical import load_contract_catalog, load_target_catalog
from codeintel.core.exports.formats import export_format_choices, resolve_export_format_spec
from codeintel.core.imports.lazy import lazy_getattr

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.targets import OutputTarget
    from codeintel.core.exports.formats import ExportFormat, ExportFormatSpec
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.serving.semantic.models import SemanticViewSpec
    from codeintel.serving.semantic.registry import SemanticRegistry
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True, slots=True)
class RegistryService:
    """Registry service for catalog and semantic discovery."""

    contract_catalog: Mapping[str, DatasetContract]
    target_catalog: Mapping[str, OutputTarget]
    semantic_registry: SemanticRegistry | None = None

    @classmethod
    def empty(cls) -> RegistryService:
        """Return an empty registry service.

        Returns
        -------
        RegistryService
            Registry service with empty catalogs.
        """
        return cls(contract_catalog={}, target_catalog={}, semantic_registry=None)

    @classmethod
    def from_gateway(
        cls,
        *,
        gateway: StorageGateway | None = None,
        root: Path | None = None,
    ) -> RegistryService:
        """Load dataset and target catalogs from canonical storage.

        Returns
        -------
        RegistryService
            Registry service populated with dataset and target catalogs.
        """
        contracts = load_contract_catalog(gateway=gateway, root=root)
        targets = load_target_catalog(gateway=gateway, root=root)
        return cls(contract_catalog=contracts, target_catalog=targets, semantic_registry=None)

    @classmethod
    def from_semantic_registry(cls, registry: SemanticRegistry) -> RegistryService:
        """Build a registry service from a semantic registry.

        Returns
        -------
        RegistryService
            Registry service with the supplied semantic registry.
        """
        return cls(contract_catalog={}, target_catalog={}, semantic_registry=registry)

    @classmethod
    def from_semantic_registry_path(cls, path: Path) -> RegistryService:
        """Load a semantic registry from disk and build the registry service.

        Returns
        -------
        RegistryService
            Registry service loaded from the semantic registry path.
        """
        registry_cls = cast(
            "type[SemanticRegistry]",
            lazy_getattr("codeintel.serving.semantic.registry", "SemanticRegistry"),
        )
        registry = registry_cls.load(path)
        return cls(contract_catalog={}, target_catalog={}, semantic_registry=registry)

    def get_contract(self, table_key: str) -> DatasetContract:
        """Return the dataset contract for a table key.

        Returns
        -------
        DatasetContract
            Dataset contract for the table key.

        Raises
        ------
        KeyError
            If the table key is not present in the contract catalog.
        """
        contract = self.contract_catalog.get(table_key)
        if contract is None:
            msg = f"Unknown dataset contract: {table_key}"
            raise KeyError(msg)
        return contract

    def iter_contracts(self) -> Iterable[DatasetContract]:
        """Iterate all dataset contracts.

        Returns
        -------
        Iterable[DatasetContract]
            Iterable of dataset contracts.
        """
        return self.contract_catalog.values()

    def get_target(self, name: str) -> OutputTarget:
        """Return the OutputTarget metadata for a target name.

        Returns
        -------
        OutputTarget
            Output target metadata for the requested name.

        Raises
        ------
        KeyError
            If the target name is not present in the catalog.
        """
        target = self.target_catalog.get(name)
        if target is None:
            msg = f"Unknown output target: {name}"
            raise KeyError(msg)
        return target

    def iter_targets(self) -> Iterable[OutputTarget]:
        """Iterate all OutputTarget metadata entries.

        Returns
        -------
        Iterable[OutputTarget]
            Iterable of output target metadata.
        """
        return self.target_catalog.values()

    def get_semantic_view(self, view_id: str) -> SemanticViewSpec:
        """Return the semantic view specification for a semantic ID.

        Returns
        -------
        SemanticViewSpec
            Semantic view specification for the requested ID.

        Raises
        ------
        KeyError
            If the semantic registry is unavailable or the view is missing.
        """
        if self.semantic_registry is None:
            msg = "Semantic registry is not available"
            raise KeyError(msg)
        return self.semantic_registry.by_id(view_id)

    def iter_semantic_views(self) -> Iterable[SemanticViewSpec]:
        """Iterate all semantic view specifications.

        Returns
        -------
        Iterable[SemanticViewSpec]
            Iterable of semantic view specifications.
        """
        if self.semantic_registry is None:
            return ()
        return self.semantic_registry.views

    @staticmethod
    def export_format_spec(fmt: str) -> ExportFormatSpec:
        """Return the export format specification for a raw format string.

        Returns
        -------
        ExportFormatSpec
            Specification for the resolved export format.
        """
        return resolve_export_format_spec(fmt)

    @staticmethod
    def export_format_choices() -> tuple[ExportFormat, ...]:
        """Return supported export formats in a stable order.

        Returns
        -------
        tuple[ExportFormat, ...]
            Ordered export format identifiers.
        """
        return export_format_choices()


__all__ = ["RegistryService"]
