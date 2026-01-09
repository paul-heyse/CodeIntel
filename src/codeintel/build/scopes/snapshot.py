"""Strict snapshot scoping helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, equal_expr, equal_mask
from codeintel.config.primitives import SnapshotRef
from codeintel.core.columnar.streaming import DatasetScanOptions
from codeintel.core.config.settings import ArrowScanSettings
from codeintel.core.runtime.loader import load_runtime_settings


@dataclass(frozen=True, slots=True)
class SnapshotScope:
    """Strict repo/commit scope for snapshot-aligned filtering."""

    repo: str
    commit: str

    @classmethod
    def from_snapshot(cls, snapshot: SnapshotRef) -> SnapshotScope:
        """Construct a scope from a snapshot reference.

        Returns
        -------
        SnapshotScope
            Snapshot scope aligned to the provided reference.
        """
        return cls(repo=snapshot.repo, commit=snapshot.commit)

    def filter_arrow_table(
        self,
        table: pa.Table,
        *,
        require_columns: bool = True,
    ) -> pa.Table:
        """Filter an Arrow table to the scoped repo/commit.

        Returns
        -------
        pa.Table
            Filtered table containing only rows for the snapshot scope.

        Raises
        ------
        ValueError
            If snapshot columns are missing and require_columns is True.
        """
        missing = [name for name in ("repo", "commit") if name not in table.column_names]
        if missing:
            if require_columns:
                msg = f"Missing snapshot columns: {missing}"
                raise ValueError(msg)
            return table
        repo_mask = equal_mask(table["repo"], pa.scalar(self.repo))
        commit_mask = equal_mask(table["commit"], pa.scalar(self.commit))
        return safe_filter(table, and_kleene(repo_mask, commit_mask))

    def filter_rows(
        self,
        rows: list[dict[str, object]],
        *,
        require_keys: bool = True,
    ) -> list[dict[str, object]]:
        """Filter row dicts to the scoped repo/commit.

        Returns
        -------
        list[dict[str, object]]
            Filtered rows matching the snapshot scope.

        Raises
        ------
        ValueError
            If snapshot keys are missing and require_keys is True.
        """
        filtered: list[dict[str, object]] = []
        for row in rows:
            if require_keys and ("repo" not in row or "commit" not in row):
                msg = "Missing snapshot keys in row"
                raise ValueError(msg)
            if "repo" in row and row.get("repo") != self.repo:
                continue
            if "commit" in row and row.get("commit") != self.commit:
                continue
            filtered.append(row)
        return filtered


@dataclass(frozen=True, slots=True)
class SnapshotScanContext:
    """Context for snapshot-aligned dataset scanning."""

    repo: str | None
    commit: str | None
    settings: ArrowScanSettings

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SnapshotRef,
        *,
        settings: ArrowScanSettings | None = None,
    ) -> SnapshotScanContext:
        """Construct a scan context from a snapshot reference.

        Returns
        -------
        SnapshotScanContext
            Scan context aligned to the snapshot.
        """
        resolved_settings = settings or load_runtime_settings().build.arrow_scan
        return cls(repo=snapshot.repo, commit=snapshot.commit, settings=resolved_settings)

    def filter_expr(self, schema: pa.Schema) -> ds.Expression | None:
        """Return a dataset filter expression for the snapshot.

        Returns
        -------
        pyarrow.dataset.Expression | None
            Filter expression for repo/commit, or None if columns are missing.
        """
        expression: ds.Expression | None = None
        if self.repo is not None and "repo" in schema.names:
            expression = equal_expr("repo", self.repo)
        if self.commit is not None and "commit" in schema.names:
            commit_expr = equal_expr("commit", self.commit)
            expression = commit_expr if expression is None else expression & commit_expr
        return expression

    def scan_options(
        self,
        *,
        columns: Sequence[str] | Mapping[str, ds.Expression] | None,
        batch_size: int | None = None,
    ) -> DatasetScanOptions:
        """Build dataset scan options using the stored settings.

        Returns
        -------
        DatasetScanOptions
            Scan options aligned to the snapshot.
        """
        return DatasetScanOptions(
            batch_size=batch_size or self.settings.batch_size,
            columns=columns,
            filter_expression=None,
            cache_metadata=self.settings.cache_metadata,
            use_threads=self.settings.use_threads,
            batch_readahead=self.settings.batch_readahead,
            fragment_readahead=self.settings.fragment_readahead,
            parquet_pre_buffer=self.settings.parquet_pre_buffer,
            parquet_use_buffered_stream=self.settings.parquet_use_buffered_stream,
            parquet_buffer_size=self.settings.parquet_buffer_size,
            unify_schemas=True,
        )


__all__ = ["SnapshotScanContext", "SnapshotScope"]
