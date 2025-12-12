"""CLI-specific seed packs for handler testing.

This module provides seed packs that prepare test data for CLI handler tests.
These packs extend the core seed packs with CLI-specific data such as
storage profiles, operation metadata, and macro definitions.

The packs follow the SeedPack protocol and can be composed with other packs
using TestContext.require().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tests._helpers.seeds.core import CORE_PACK
from tests._helpers.seeds.graph import GRAPH_PACK
from tests._helpers.seeds.subsystems import SUBSYSTEM_PACK

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


@dataclass
class CliCorePack:
    """Seed pack extending CORE_PACK with CLI-specific metadata.

    Prepare the minimal environment needed for CLI handler tests,
    including repository mapping and basic module data.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    operation_id_prefix : str
        Prefix for operation IDs in tests.
    """

    name: str = "cli_core"
    operation_id_prefix: str = "cli.test"
    _dependencies: tuple[SeedPack, ...] = field(default_factory=lambda: (CORE_PACK,))

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for base data.
        """
        return self._dependencies

    def apply(self, ctx: TestContext) -> None:
        """Apply CLI core seeds to the test context.

        Parameters
        ----------
        ctx
            Test context to seed.
        """


@dataclass
class StorageProfilePack:
    """Seed pack for storage profile and macro data.

    Prepare data needed for storage handler tests including macro
    definitions and profile metadata.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_sample_macros : bool
        Whether to seed sample macro definitions.
    """

    name: str = "storage_profile"
    include_sample_macros: bool = True
    _dependencies: tuple[SeedPack, ...] = field(default_factory=lambda: (CORE_PACK,))

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for base data.
        """
        return self._dependencies

    def apply(self, ctx: TestContext) -> None:
        """Apply storage profile seeds to the test context.

        Parameters
        ----------
        ctx
            Test context to seed.
        """


@dataclass
class OperationRegistryPack:
    """Seed pack for operation registry metadata.

    Prepare data needed for operation handler tests.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    """

    name: str = "operation_registry"
    _dependencies: tuple[SeedPack, ...] = field(default_factory=lambda: (CORE_PACK,))

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for base data.
        """
        return self._dependencies

    def apply(self, ctx: TestContext) -> None:
        """Apply operation registry seeds to the test context.

        Parameters
        ----------
        ctx
            Test context to seed.
        """


@dataclass
class GraphHandlerPack:
    """Seed pack for graph handler tests.

    Combine CORE_PACK and GRAPH_PACK with any additional data
    needed for graph handler testing.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    """

    name: str = "graph_handler"
    _dependencies: tuple[SeedPack, ...] = field(default_factory=lambda: (CORE_PACK, GRAPH_PACK))

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack and GraphPack are required.
        """
        return self._dependencies

    def apply(self, ctx: TestContext) -> None:
        """Apply graph handler seeds to the test context.

        Parameters
        ----------
        ctx
            Test context to seed.
        """


@dataclass
class SubsystemHandlerPack:
    """Seed pack for subsystem handler tests.

    Combine CORE_PACK and SUBSYSTEM_PACK with any additional data
    needed for subsystem handler testing.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    """

    name: str = "subsystem_handler"
    _dependencies: tuple[SeedPack, ...] = field(default_factory=lambda: (CORE_PACK, SUBSYSTEM_PACK))

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack and SubsystemPack are required.
        """
        return self._dependencies

    def apply(self, ctx: TestContext) -> None:
        """Apply subsystem handler seeds to the test context.

        Parameters
        ----------
        ctx
            Test context to seed.
        """


CLI_CORE_PACK = CliCorePack()
STORAGE_PROFILE_PACK = StorageProfilePack()
OPERATION_REGISTRY_PACK = OperationRegistryPack()
GRAPH_HANDLER_PACK = GraphHandlerPack()
SUBSYSTEM_HANDLER_PACK = SubsystemHandlerPack()


__all__ = [
    "CLI_CORE_PACK",
    "GRAPH_HANDLER_PACK",
    "OPERATION_REGISTRY_PACK",
    "STORAGE_PROFILE_PACK",
    "SUBSYSTEM_HANDLER_PACK",
    "CliCorePack",
    "GraphHandlerPack",
    "OperationRegistryPack",
    "StorageProfilePack",
    "SubsystemHandlerPack",
]
