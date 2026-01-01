"""Materializers and I/O adapters for Hamilton build graphs.

This package provides Hamilton-native I/O building blocks (DataSavers/DataLoaders)
that make side effects (Parquet writes, file outputs) visible in the Hamilton DAG.
"""

from __future__ import annotations

from codeintel.build.hamilton.materializers.arrow_dataset_saver import ArrowDatasetSaver
from codeintel.build.hamilton.materializers.artifact_saver import FileArtifactSaver

__all__ = [
    "ArrowDatasetSaver",
    "FileArtifactSaver",
]
