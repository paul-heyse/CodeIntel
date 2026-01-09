"""Query specification helpers for Arrow dataset scans and Acero plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow.compute as pc
import pyarrow.dataset as ds

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

    def scan_columns(
        self,
        *,
        provenance: bool,
    ) -> Sequence[str] | Mapping[str, pc.Expression] | None:
        """Return scan columns with optional provenance and computed expressions.

        Returns
        -------
        Sequence[str] | Mapping[str, pc.Expression] | None
            Column list or mapping for scan node configuration.
        """
        return _scan_columns_from_projection(
            self.base_cols,
            self.computed,
            provenance=provenance,
        )

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

    def scan_columns(
        self,
        *,
        provenance: bool,
    ) -> Sequence[str] | Mapping[str, pc.Expression] | None:
        """Return scan columns for dataset or Acero scan nodes.

        Returns
        -------
        Sequence[str] | Mapping[str, pc.Expression] | None
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

    def scan_filter_expression(self) -> pc.Expression | None:
        """Return the scan filter expression (pushdown preferred).

        Returns
        -------
        pyarrow.compute.Expression | None
            Filter expression for scan nodes.
        """
        return self.pushdown_predicate or self.predicate

    def post_filter_expression(self) -> pc.Expression | None:
        """Return the post-scan filter expression.

        Returns
        -------
        pyarrow.compute.Expression | None
            Filter expression applied after scan nodes.
        """
        if self.predicate is None:
            return None
        if self.pushdown_predicate is self.predicate:
            return None
        return self.predicate


def projection_spec_from_columns(
    columns: Sequence[str] | Mapping[str, pc.Expression | ds.Expression] | None,
    *,
    default_columns: Sequence[str] | None = None,
    provenance_columns: Sequence[str] = (),
) -> ProjectionSpec:
    """Build a ProjectionSpec from base or computed columns.

    Parameters
    ----------
    columns
        Column selection or computed expressions.
    default_columns
        Default columns to include when ``columns`` is None.
    provenance_columns
        Additional provenance columns to include in the projection.

    Returns
    -------
    ProjectionSpec
        Projection spec composed from provided columns.
    """
    base_cols: list[str] = []
    computed: list[tuple[str, pc.Expression]] = []
    if columns is None:
        if default_columns is not None:
            base_cols = list(default_columns)
    elif isinstance(columns, Mapping):
        computed = [(name, cast("pc.Expression", expr)) for name, expr in columns.items()]
    else:
        base_cols = list(columns)
    computed_names = {name for name, _expr in computed}
    for name in provenance_columns:
        if name in computed_names or name in base_cols:
            continue
        base_cols.append(name)
    return ProjectionSpec(base_cols=tuple(base_cols), computed=tuple(computed))


def _scan_columns_from_projection(
    base_cols: Sequence[str],
    computed: Sequence[tuple[str, pc.Expression]],
    *,
    provenance: bool,
) -> Sequence[str] | Mapping[str, pc.Expression] | None:
    needs_mapping = bool(computed) or provenance
    if not needs_mapping:
        if not base_cols:
            return None
        return list(base_cols)
    columns: dict[str, pc.Expression] = {col: E.field(col) for col in base_cols}
    if computed:
        columns.update(dict(computed))
    if provenance:
        columns.update(
            {
                output_name: E.field(source_name)
                for output_name, source_name in PROVENANCE_FIELDS
            }
        )
    return columns


__all__ = [
    "PROVENANCE_FIELDS",
    "ProjectionSpec",
    "QuerySpec",
    "projection_spec_from_columns",
]
