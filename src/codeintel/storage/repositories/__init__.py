"""Repository layer for DuckDB persistence.

This package provides repository classes for querying DuckDB tables with
snapshot (repo/commit) scoping. Use RepositoryFactory for convenient
creation of snapshot-scoped repositories.

Example
-------
>>> from codeintel.storage.repositories import RepositoryFactory
>>>
>>> factory = RepositoryFactory(gateway, repo="org/repo", commit="abc123")
>>> summary = factory.functions.get_function_summary_by_goid(goid)
"""

from codeintel.storage.repositories.base import (
    BaseRepository,
    PaginatedRows,
    RowDict,
)
from codeintel.storage.repositories.data_models import (
    DataModelFieldRow,
    DataModelRelationshipRow,
    DataModelRow,
    NormalizedDataModel,
    fetch_fields,
    fetch_models,
    fetch_models_normalized,
    fetch_relationships,
)
from codeintel.storage.repositories.dataflow import DataflowRepository
from codeintel.storage.repositories.datasets import DatasetReadRepository
from codeintel.storage.repositories.factory import RepositoryFactory
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.graphs import GraphRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.repositories.tests import TestRepository

__all__ = [
    "BaseRepository",
    "DataModelFieldRow",
    "DataModelRelationshipRow",
    "DataModelRow",
    "DataflowRepository",
    "DatasetReadRepository",
    "FunctionRepository",
    "GraphRepository",
    "ModuleRepository",
    "NormalizedDataModel",
    "PaginatedRows",
    "RepositoryFactory",
    "RowDict",
    "SubsystemRepository",
    "TestRepository",
    "fetch_fields",
    "fetch_models",
    "fetch_models_normalized",
    "fetch_relationships",
]
