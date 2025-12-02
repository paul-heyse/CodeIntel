"""Repository layer for DuckDB persistence."""

from codeintel.storage.repositories.base import (
    BaseRepository,
    PaginatedRows,
    RowDict,
    fetch_all_dicts,
    fetch_one_dict,
    fetch_paginated,
    row_exists,
)
from codeintel.storage.repositories.data_models import DataModelRepository
from codeintel.storage.repositories.dataflow import DataflowRepository
from codeintel.storage.repositories.datasets import DatasetReadRepository
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.graphs import GraphRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.repositories.tests import TestRepository

__all__ = [
    "BaseRepository",
    "DataModelRepository",
    "DataflowRepository",
    "DatasetReadRepository",
    "FunctionRepository",
    "GraphRepository",
    "ModuleRepository",
    "PaginatedRows",
    "RowDict",
    "SubsystemRepository",
    "TestRepository",
    "fetch_all_dicts",
    "fetch_one_dict",
    "fetch_paginated",
    "row_exists",
]
