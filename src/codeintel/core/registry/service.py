"""Canonical registry service for datasets, targets, and semantic views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from codeintel.build.target_metadata import get_target_system
from codeintel.core.exports.formats import export_format_choices, resolve_export_format_spec
from codeintel.core.imports.lazy import lazy_getattr

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.schemas.contract_service import ContractService
    from codeintel.build.targets import OutputTarget
    from codeintel.core.exports.formats import ExportFormat, ExportFormatSpec
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.serving.semantic.models import SemanticViewSpec
    from codeintel.serving.semantic.registry import SemanticRegistry
    from codeintel.storage.gateway import StorageGateway


_DAG_OUTPUT_INVENTORY_PATH = Path(__file__).with_name("dag_output_inventory.yaml")
_VALID_MATERIALIZATIONS = {"artifact", "mixed", "table"}


class RegistryTypeError(TypeError):
    """Type errors raised while validating registry payloads."""

    def __init__(self, message: str) -> None:
        super().__init__(message)

    @classmethod
    def expected(cls, field: str, expected: str) -> RegistryTypeError:
        """Build a type error for an unexpected field type.

        Returns
        -------
        RegistryTypeError
            Constructed type error instance.
        """
        message = f"Expected '{field}' to be {expected}."
        return cls(message)

    @classmethod
    def expected_mapping_keys(cls, field: str) -> RegistryTypeError:
        """Build a type error for unexpected mapping keys.

        Returns
        -------
        RegistryTypeError
            Constructed type error instance.
        """
        message = f"Expected '{field}' keys to be strings."
        return cls(message)

    @classmethod
    def expected_list_items(cls, field: str) -> RegistryTypeError:
        """Build a type error for unexpected list items.

        Returns
        -------
        RegistryTypeError
            Constructed type error instance.
        """
        message = f"Expected '{field}' to contain only strings."
        return cls(message)


class RegistryValidationError(ValueError):
    """Value errors raised while validating registry payloads."""

    def __init__(self, message: str) -> None:
        super().__init__(message)

    @classmethod
    def unsupported_materialization(cls, materialization: str) -> RegistryValidationError:
        """Build a validation error for unsupported materializations.

        Returns
        -------
        RegistryValidationError
            Constructed validation error instance.
        """
        message = f"Unsupported materialization '{materialization}'."
        return cls(message)

    @classmethod
    def missing_table_keys(cls, target: str) -> RegistryValidationError:
        """Build a validation error for missing table keys.

        Returns
        -------
        RegistryValidationError
            Constructed validation error instance.
        """
        message = f"Output '{target}' must define table_keys."
        return cls(message)

    @classmethod
    def unexpected_table_keys(cls, target: str) -> RegistryValidationError:
        """Build a validation error for unexpected table keys.

        Returns
        -------
        RegistryValidationError
            Constructed validation error instance.
        """
        message = f"Output '{target}' must not define table_keys for artifacts."
        return cls(message)

    @classmethod
    def load_failed(cls, path: Path) -> RegistryValidationError:
        """Build a validation error for failed inventory loads.

        Returns
        -------
        RegistryValidationError
            Constructed validation error instance.
        """
        message = f"Failed to load DAG output inventory: {path}"
        return cls(message)

    @classmethod
    def inventory_empty(cls, path: Path) -> RegistryValidationError:
        """Build a validation error for empty inventory files.

        Returns
        -------
        RegistryValidationError
            Constructed validation error instance.
        """
        message = f"DAG output inventory is empty: {path}"
        return cls(message)

    @classmethod
    def outputs_not_list(cls) -> RegistryValidationError:
        """Build a validation error for invalid outputs payloads.

        Returns
        -------
        RegistryValidationError
            Constructed validation error instance.
        """
        message = "Expected 'outputs' to be a list of mappings."
        return cls(message)

    @classmethod
    def duplicate_output(cls, target: str) -> RegistryValidationError:
        """Build a validation error for duplicate output targets.

        Returns
        -------
        RegistryValidationError
            Constructed validation error instance.
        """
        message = f"Duplicate output target '{target}'."
        return cls(message)


class RegistryLookupError(KeyError):
    """Lookup errors for registry entities."""

    def __init__(self, message: str) -> None:
        super().__init__(message)

    @classmethod
    def missing_output(cls, target: str) -> RegistryLookupError:
        """Build a lookup error for missing output targets.

        Returns
        -------
        RegistryLookupError
            Constructed lookup error instance.
        """
        message = f"Unknown DAG output target: {target}"
        return cls(message)

    @classmethod
    def missing_contract(cls, table_key: str) -> RegistryLookupError:
        """Build a lookup error for missing dataset contracts.

        Returns
        -------
        RegistryLookupError
            Constructed lookup error instance.
        """
        message = f"Unknown dataset contract: {table_key}"
        return cls(message)

    @classmethod
    def missing_target(cls, name: str) -> RegistryLookupError:
        """Build a lookup error for missing output targets.

        Returns
        -------
        RegistryLookupError
            Constructed lookup error instance.
        """
        message = f"Unknown output target: {name}"
        return cls(message)

    @classmethod
    def semantic_registry_unavailable(cls) -> RegistryLookupError:
        """Build a lookup error for missing semantic registries.

        Returns
        -------
        RegistryLookupError
            Constructed lookup error instance.
        """
        message = "Semantic registry is not available"
        return cls(message)


def _require_str(value: object, *, field: str) -> str:
    if isinstance(value, str):
        return value
    raise RegistryTypeError.expected(field, "a string")


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, int):
        return value
    raise RegistryTypeError.expected(field, "an integer")


def _require_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise RegistryTypeError.expected(field, "a boolean")


def _optional_str(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RegistryTypeError.expected(field, "a string or null")


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RegistryTypeError.expected(field, "a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RegistryTypeError.expected_mapping_keys(field)
        result[key] = item
    return result


def _optional_str_list(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RegistryTypeError.expected(field, "a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RegistryTypeError.expected_list_items(field)
        items.append(item)
    return tuple(items)


@dataclass(frozen=True, slots=True)
class DagOutputSpec:
    """Structured output metadata for DAG inventory entries."""

    target: str
    domain: str
    anchor: str
    materialization: str
    table_keys: tuple[str, ...]
    contracts: tuple[str, ...]
    downstream_consumers: tuple[str, ...]
    upstream_targets: tuple[str, ...]
    pilot: bool = False
    notes: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> DagOutputSpec:
        """Parse and validate a DAG output spec mapping.

        Parameters
        ----------
        raw
            Raw mapping from the inventory payload.

        Returns
        -------
        DagOutputSpec
            Parsed DAG output specification.

        Raises
        ------
        RegistryValidationError.unsupported_materialization
            If the materialization value is not supported.
        RegistryValidationError.missing_table_keys
            If table materializations omit table keys.
        RegistryValidationError.unexpected_table_keys
            If artifact materializations define table keys.
        """
        target = _require_str(raw.get("target"), field="target")
        domain = _require_str(raw.get("domain"), field="domain")
        anchor = _require_str(raw.get("anchor"), field="anchor")
        materialization = _require_str(raw.get("materialization"), field="materialization")
        if materialization not in _VALID_MATERIALIZATIONS:
            raise RegistryValidationError.unsupported_materialization(materialization)

        table_keys = _optional_str_list(raw.get("table_keys"), field="table_keys")
        contracts = _optional_str_list(raw.get("contracts"), field="contracts")
        downstream = _optional_str_list(
            raw.get("downstream_consumers"),
            field="downstream_consumers",
        )
        upstream = _optional_str_list(raw.get("upstream_targets"), field="upstream_targets")
        pilot = raw.get("pilot", False)
        notes = _optional_str(raw.get("notes"), field="notes")

        if not isinstance(pilot, bool):
            pilot = _require_bool(pilot, field="pilot")

        if materialization == "table" and not table_keys:
            raise RegistryValidationError.missing_table_keys(target)
        if materialization == "artifact" and table_keys:
            raise RegistryValidationError.unexpected_table_keys(target)

        if not contracts and table_keys:
            contracts = table_keys

        return cls(
            target=target,
            domain=domain,
            anchor=anchor,
            materialization=materialization,
            table_keys=table_keys,
            contracts=contracts,
            downstream_consumers=downstream,
            upstream_targets=upstream,
            pilot=pilot,
            notes=notes,
        )


@dataclass(frozen=True, slots=True)
class DagOutputInventory:
    """Inventory for DAG outputs with validation helpers."""

    version: int
    generated_at: str | None
    outputs: tuple[DagOutputSpec, ...]

    @classmethod
    def from_path(cls, path: Path) -> DagOutputInventory:
        """Load and validate a DAG output inventory file.

        Parameters
        ----------
        path
            Path to the inventory YAML file.

        Returns
        -------
        DagOutputInventory
            Parsed inventory data.

        Raises
        ------
        RegistryValidationError.load_failed
            If the inventory file cannot be loaded.
        RegistryValidationError.inventory_empty
            If the inventory file is empty.
        RegistryValidationError.outputs_not_list
            If the outputs payload is not a list.
        RegistryValidationError.duplicate_output
            If output targets are duplicated.
        """
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RegistryValidationError.load_failed(path) from exc

        if raw is None:
            raise RegistryValidationError.inventory_empty(path)

        payload = _require_mapping(raw, field="inventory")
        version = _require_int(payload.get("version"), field="version")
        generated_at = _optional_str(payload.get("generated_at"), field="generated_at")

        outputs_raw = payload.get("outputs")
        if not isinstance(outputs_raw, list):
            raise RegistryValidationError.outputs_not_list()

        outputs: list[DagOutputSpec] = []
        seen: set[str] = set()
        for item in outputs_raw:
            output_raw = _require_mapping(item, field="outputs")
            spec = DagOutputSpec.from_mapping(output_raw)
            if spec.target in seen:
                raise RegistryValidationError.duplicate_output(spec.target)
            seen.add(spec.target)
            outputs.append(spec)

        return cls(version=version, generated_at=generated_at, outputs=tuple(outputs))

    def by_target(self, target: str) -> DagOutputSpec:
        """Return the output spec for a target name.

        Parameters
        ----------
        target
            Output target name to resolve.

        Returns
        -------
        DagOutputSpec
            Output spec for the target.

        Raises
        ------
        RegistryLookupError.missing_output
            If the target is not present in the inventory.
        """
        for spec in self.outputs:
            if spec.target == target:
                return spec
        raise RegistryLookupError.missing_output(target)

    def iter_outputs(self) -> Iterable[DagOutputSpec]:
        """Iterate all output specs.

        Returns
        -------
        Iterable[DagOutputSpec]
            Output specifications in the inventory.
        """
        return self.outputs


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
        """Load dataset and target catalogs from the build graph.

        Returns
        -------
        RegistryService
            Registry service populated with dataset and target catalogs.
        """
        _ = gateway
        _ = root
        contract_service_factory = cast(
            "Callable[[], ContractService]",
            lazy_getattr(
                "codeintel.build.schemas.contract_service",
                "get_enriched_contract_service",
            ),
        )
        contracts = {
            contract.table_key: contract
            for contract in contract_service_factory().iter_contracts()
        }
        targets = {target.name: target for target in get_target_system().graph.all_targets}
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

    @staticmethod
    def load_dag_output_inventory(path: Path | None = None) -> DagOutputInventory:
        """Load the DAG output inventory from disk.

        Parameters
        ----------
        path
            Optional path override for the inventory file.

        Returns
        -------
        DagOutputInventory
            Parsed DAG output inventory.
        """
        inventory_path = path or _DAG_OUTPUT_INVENTORY_PATH
        return DagOutputInventory.from_path(inventory_path)

    @staticmethod
    def default_dag_output_inventory_path() -> Path:
        """Return the canonical DAG output inventory path.

        Returns
        -------
        Path
            Default inventory file path.
        """
        return _DAG_OUTPUT_INVENTORY_PATH

    def get_contract(self, table_key: str) -> DatasetContract:
        """Return the dataset contract for a table key.

        Returns
        -------
        DatasetContract
            Dataset contract for the table key.

        Raises
        ------
        RegistryLookupError.missing_contract
            If the table key is not present in the contract catalog.
        """
        contract = self.contract_catalog.get(table_key)
        if contract is None:
            raise RegistryLookupError.missing_contract(table_key)
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
        RegistryLookupError.missing_target
            If the target name is not present in the catalog.
        """
        target = self.target_catalog.get(name)
        if target is None:
            raise RegistryLookupError.missing_target(name)
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
        RegistryLookupError.semantic_registry_unavailable
            If the semantic registry is unavailable or the view is missing.
        """
        if self.semantic_registry is None:
            raise RegistryLookupError.semantic_registry_unavailable()
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


__all__ = [
    "DagOutputInventory",
    "DagOutputSpec",
    "RegistryLookupError",
    "RegistryService",
    "RegistryTypeError",
    "RegistryValidationError",
]
