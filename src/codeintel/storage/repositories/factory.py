"""Factory for creating snapshot-scoped repositories.

This module provides a unified factory for creating repository instances
bound to a specific repo/commit snapshot. This ensures consistent
initialization and enables lazy repository creation.

Example
-------
>>> from codeintel.storage.repositories import RepositoryFactory
>>>
>>> from codeintel.core.storage import StorageContext
>>> factory = RepositoryFactory(StorageContext(gateway=gateway, snapshot=snapshot))
>>> functions = factory.functions
>>> modules = factory.modules
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from codeintel.core.storage import StorageContext
from codeintel.storage.repositories.data_models import DataModelsRepository
from codeintel.storage.repositories.dataflow import DataflowRepository
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.graphs import GraphRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.repositories.tests import TestRepository

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["RepositoryFactory"]


class RepositoryFactory:
    """Factory for creating snapshot-scoped repositories.

    Create repository instances lazily and cache them for reuse.
    All repositories share the same gateway/repo/commit binding.

    Parameters
    ----------
    context
        Storage context providing gateway access and snapshot identity.

    Examples
    --------
    >>> factory = RepositoryFactory(StorageContext(gateway=gateway, snapshot=snapshot))
    >>> architecture = factory.functions.get_function_architecture(goid)
    """

    def __init__(self, context: StorageContext) -> None:
        """Initialize the repository factory.

        Parameters
        ----------
        context
            Storage context providing gateway access and snapshot identity.
        """
        self._context = context

    @property
    def context(self) -> StorageContext:
        """Return the underlying storage context."""
        return self._context

    @property
    def gateway(self) -> StorageGateway:
        """Return the underlying storage gateway."""
        return self._context.gateway

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference."""
        return self._context.require_snapshot()

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        return self._context.repo

    @property
    def commit(self) -> str:
        """Return the commit hash."""
        return self._context.commit

    @cached_property
    def functions(self) -> FunctionRepository:
        """Return the function repository.

        Returns
        -------
        FunctionRepository
            Repository for function-centric queries.
        """
        return FunctionRepository(self._context)

    @cached_property
    def modules(self) -> ModuleRepository:
        """Return the module repository.

        Returns
        -------
        ModuleRepository
            Repository for module and file queries.
        """
        return ModuleRepository(self._context)

    @cached_property
    def graphs(self) -> GraphRepository:
        """Return the graph repository.

        Returns
        -------
        GraphRepository
            Repository for graph-related queries.
        """
        return GraphRepository(self._context)

    @cached_property
    def tests(self) -> TestRepository:
        """Return the test repository.

        Returns
        -------
        TestRepository
            Repository for test-related queries.
        """
        return TestRepository(self._context)

    @cached_property
    def subsystems(self) -> SubsystemRepository:
        """Return the subsystem repository.

        Returns
        -------
        SubsystemRepository
            Repository for subsystem queries.
        """
        return SubsystemRepository(self._context)

    @cached_property
    def dataflow(self) -> DataflowRepository:
        """Return the dataflow repository.

        Returns
        -------
        DataflowRepository
            Repository for dataflow queries.
        """
        return DataflowRepository(self._context)

    @cached_property
    def data_models(self) -> DataModelsRepository:
        """Return the data models repository.

        Returns
        -------
        DataModelsRepository
            Repository for data model tables and normalized views.
        """
        return DataModelsRepository(self._context)
