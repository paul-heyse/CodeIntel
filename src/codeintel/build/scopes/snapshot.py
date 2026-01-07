"""Strict snapshot scoping helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, equal_mask
from codeintel.config.primitives import SnapshotRef


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


__all__ = ["SnapshotScope"]
