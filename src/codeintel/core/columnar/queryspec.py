"""Query specification helpers for Arrow dataset scans and Acero plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pyarrow.compute as pc

from codeintel.core.columnar.expr_vocab import E

PROVENANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("prov_filename", "__filename"),
    ("prov_fragment_index", "__fragment_index"),
    ("prov_batch_index", "__batch_index"),
    ("prov_last_in_fragment", "__last_in_fragment"),
)


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Projection details for QuerySpec compilation."""

    base_cols: tuple[str, ...]
    computed: tuple[tuple[str, pc.Expression], ...] = ()

    def scan_columns(self, *, provenance: bool) -> list[str] | Mapping[str, pc.Expression]:
        """Return scan columns with optional provenance and computed expressions.

        Returns
        -------
        list[str] | Mapping[str, pc.Expression]
            Column list or mapping for scan node configuration.
        """
        needs_mapping = bool(self.computed) or provenance
        if not needs_mapping:
            return list(self.base_cols)
        columns: dict[str, pc.Expression] = {col: E.field(col) for col in self.base_cols}
        if self.computed:
            columns.update(dict(self.computed))
        if provenance:
            columns.update(
                {
                    output_name: E.field(source_name)
                    for output_name, source_name in PROVENANCE_FIELDS
                }
            )
        return columns

    def project_expressions(self, *, provenance: bool) -> Mapping[str, pc.Expression]:
        """Return projection expressions for plan project nodes.

        Returns
        -------
        Mapping[str, pc.Expression]
            Mapping of output column names to expressions.
        """
        columns: dict[str, pc.Expression] = {col: E.field(col) for col in self.base_cols}
        if self.computed:
            columns.update(dict(self.computed))
        if provenance:
            columns.update(
                {
                    output_name: E.field(output_name)
                    for output_name, _source_name in PROVENANCE_FIELDS
                }
            )
        return columns


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """Query specification for scan and plan compilation."""

    predicate: pc.Expression | None
    pushdown_predicate: pc.Expression | None
    projection: ProjectionSpec

    def scan_columns(self, *, provenance: bool) -> list[str] | Mapping[str, pc.Expression]:
        """Return scan columns for dataset or Acero scan nodes.

        Returns
        -------
        list[str] | Mapping[str, pc.Expression]
            Column list or mapping for scan node configuration.
        """
        return self.projection.scan_columns(provenance=provenance)

    def project_expressions(self, *, provenance: bool) -> Mapping[str, pc.Expression]:
        """Return projection expressions for plan project nodes.

        Returns
        -------
        Mapping[str, pc.Expression]
            Mapping of output column names to expressions.
        """
        return self.projection.project_expressions(provenance=provenance)


__all__ = ["PROVENANCE_FIELDS", "ProjectionSpec", "QuerySpec"]
