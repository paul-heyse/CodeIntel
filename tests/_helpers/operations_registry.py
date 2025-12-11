"""Helpers for building operation-to-target mappings without mutating globals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.operations import OperationTargets

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.targets import OutputTarget
    from codeintel.serving.operations.catalog import Operation


@dataclass(frozen=True)
class OperationRegistryBuilder:
    """Construct table/graph target indexes for operations in tests.

    This avoids mutating the private caches in ``codeintel.build.operations``
    by computing the same mappings from explicit target lists.
    """

    targets: Sequence[OutputTarget]
    graph_mapping: Mapping[str, str] | None = None

    def table_index(self) -> dict[str, str]:
        """Build mapping from table key to target name.

        Returns
        -------
        dict[str, str]
            Mapping from table_key to target name.
        """
        index: dict[str, str] = {}
        for target in self.targets:
            for table in target.tables:
                index[table] = target.name
        return index

    def graph_index(self) -> dict[str, str]:
        """Build mapping from graph runtime name to target name.

        Returns
        -------
        dict[str, str]
            Mapping from graph runtime name to target name.
        """
        if self.graph_mapping is not None:
            return dict(self.graph_mapping)
        # Default mirrors production mapping for core graphs.
        return {"callgraph": "call_graph", "importgraph": "import_graph"}

    def build_targets_for_operation(self, operation: Operation) -> OperationTargets:
        """Build OperationTargets for a given operation without global state.

        Returns
        -------
        OperationTargets
            Resolved targets for the provided operation.
        """
        table_to_target = self.table_index()
        graph_to_target = self.graph_index()

        data_targets = {
            table_to_target[table]
            for table in operation.required_datasets
            if table in table_to_target
        }
        graph_targets = {
            graph_to_target[graph]
            for graph in operation.required_graphs
            if graph in graph_to_target
        }
        required_targets = frozenset(data_targets | graph_targets)

        return OperationTargets(
            operation_id=operation.id,
            required_targets=required_targets,
            graph_targets=frozenset(graph_targets),
            data_targets=frozenset(data_targets),
        )


__all__ = ["OperationRegistryBuilder"]
