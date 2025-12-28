"""Polars-based semantic query engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.serving.semantic.engines.protocol import (
    EngineContext,
    ExecutablePlan,
    QueryExplain,
)
from codeintel.serving.semantic.polars_query_builder import (
    PolarsQueryBuilderError,
    apply_query_spec,
    can_apply_query_spec,
)
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.view_registry import ViewInputs

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from polars import DataFrame, LazyFrame

    from codeintel.serving.semantic.datasets import DatasetManifestEntry

    type PolarsDataFrame = DataFrame
    type PolarsLazyFrame = LazyFrame
else:
    type PolarsDataFrame = object
    type PolarsLazyFrame = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


@dataclass(frozen=True, slots=True)
class PolarsExecutablePlan:
    """Executable Polars plan wrapper."""

    lazyframe: PolarsLazyFrame
    explain_plan: str | None = None

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return an Arrow RecordBatchReader for the plan results.

        Parameters
        ----------
        batch_size
            Max batches per chunk in the returned reader.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the plan output.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        batches = self.lazyframe.collect_batches(
            chunk_size=batch_size,
            engine="streaming",
        )
        schema = self.lazyframe.collect_schema().to_arrow()
        record_batches = _record_batches_from_frames(batches)
        return pa.RecordBatchReader.from_batches(schema, record_batches)

    def to_table(self) -> pa.Table:
        """Execute the plan and return a fully materialized Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized Arrow table.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        df = self.lazyframe.collect(engine="streaming")
        return df.to_arrow()

    def explain(self) -> QueryExplain:
        """Return the Polars explain plan.

        Returns
        -------
        QueryExplain
            Explain payload with the Polars plan text.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        plan = self.explain_plan or self.lazyframe.explain(optimized=True)
        return QueryExplain(sql=None, plan=plan)

    def cleanup(self) -> None:
        """Release temporary resources after execution."""
        if self.explain_plan is not None:
            return


def _record_batches_from_frames(
    frames: Iterable[PolarsDataFrame],
) -> Iterator[pa.RecordBatch]:
    for frame in frames:
        table = frame.to_arrow()
        yield from table.to_batches()


@dataclass(frozen=True, slots=True)
class PolarsQueryEngine:
    """Polars query engine for semantic specs."""

    name: str = "polars"

    def can_run(self, spec: SemanticQuerySpec, *, ctx: EngineContext) -> bool:
        """Return True when Polars can satisfy the spec.

        Parameters
        ----------
        spec
            Semantic query spec to evaluate.
        ctx
            Engine context with view and dataset registries.

        Returns
        -------
        bool
            True if Polars can execute the spec.
        """
        if pl is None or self.name.lower() != "polars":
            return False
        if not can_apply_query_spec(
            spec=spec,
            allowed_columns=spec.allowed_columns,
            column_types=spec.column_types,
        ):
            return False
        if ctx.view_registry.get(spec.table_key) is not None:
            return True
        return ctx.dataset_manifests.get(spec.table_key) is not None

    def compile(self, spec: SemanticQuerySpec, *, ctx: EngineContext) -> ExecutablePlan:
        """Compile a semantic query spec into a Polars execution plan.

        Parameters
        ----------
        spec
            Semantic query spec to compile.
        ctx
            Engine context with data sources.

        Returns
        -------
        ExecutablePlan
            Executable Polars plan wrapper.

        Raises
        ------
        PolarsQueryBuilderError
            If Polars is unavailable or the spec is invalid.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        source = self._resolve_source(spec, ctx=ctx)
        try:
            lazyframe = apply_query_spec(
                source,
                spec=spec,
                allowed_columns=spec.allowed_columns,
                column_types=spec.column_types,
            )
        except PolarsQueryBuilderError:
            raise
        except Exception as exc:  # pragma: no cover
            msg = f"Failed to build Polars query for {spec.table_key}"
            raise PolarsQueryBuilderError(msg) from exc
        return PolarsExecutablePlan(lazyframe=lazyframe)

    def _resolve_source(self, spec: SemanticQuerySpec, *, ctx: EngineContext) -> PolarsLazyFrame:
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        view_spec = ctx.view_registry.get(spec.table_key)
        if view_spec is not None:
            inputs = ViewInputs(loader=lambda key, idx: self._scan_table(ctx, key, idx))
            lazyframe = view_spec.builder(inputs)
            if not isinstance(lazyframe, pl.LazyFrame):
                msg = f"View builder for {spec.table_key} did not return a LazyFrame"
                raise PolarsQueryBuilderError(msg)
            return lazyframe
        entry = ctx.dataset_manifests.get(spec.table_key)
        if entry is None:
            msg = f"No dataset manifest found for {spec.table_key}"
            raise PolarsQueryBuilderError(msg)
        return self._scan_entry(entry)

    @staticmethod
    def _scan_entry(entry: DatasetManifestEntry) -> PolarsLazyFrame:
        if pl is None:  # pragma: no cover
            msg = "polars is required for Polars query execution"
            raise PolarsQueryBuilderError(msg)
        if entry.manifest.files:
            paths = [str(entry.dataset_dir / path) for path in entry.manifest.files]
            return pl.scan_parquet(paths)
        glob = str(entry.dataset_dir / "**" / "*.parquet")
        return pl.scan_parquet(glob)

    def _scan_table(
        self,
        ctx: EngineContext,
        table_key: str,
        row_index: str | None,
    ) -> PolarsLazyFrame:
        entry = ctx.dataset_manifests.get(table_key)
        if entry is None:
            msg = f"No dataset manifest found for {table_key}"
            raise PolarsQueryBuilderError(msg)
        lazyframe = self._scan_entry(entry)
        if row_index:
            lazyframe = lazyframe.with_row_index(name=row_index)
        return lazyframe


__all__ = ["PolarsExecutablePlan", "PolarsQueryEngine"]
