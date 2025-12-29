"""Result builders for Hamilton execution outputs."""

from __future__ import annotations

from dataclasses import dataclass

from hamilton.lifecycle import ResultBuilder

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.run_records import TargetRunRecord


@dataclass(frozen=True, slots=True)
class BuildResultBuilder(ResultBuilder):
    """Build a filtered result mapping for Hamilton execution outputs.

    Attributes
    ----------
    allowed_nodes
        Node names to keep in the final result mapping. Nodes outside this list
        are dropped unless they match preserved value types.
    """

    allowed_nodes: tuple[str, ...]
    preserved_types: tuple[type[object], ...] = (TargetRunRecord, MaterializationResult)

    def build_result(self, **outputs: object) -> dict[str, object]:
        """Return filtered execution outputs.

        Parameters
        ----------
        **outputs
            Raw Hamilton outputs keyed by node name.

        Returns
        -------
        dict[str, object]
            Filtered mapping containing allowed nodes or preserved types.
        """
        allowed = set(self.allowed_nodes)
        return {
            name: value
            for name, value in outputs.items()
            if name in allowed or isinstance(value, self.preserved_types)
        }


__all__ = ["BuildResultBuilder"]
