"""Repository for file and module queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.core.ibis_typing import and_predicates
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


@dataclass(frozen=True)
class ModuleRepository(BaseRepository):
    """Read module and file metadata from docs views."""

    _FILE_SUMMARY_COLUMNS: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "rel_path",
        "module",
        "language",
        "function_count",
        "class_count",
        "loc",
        "complexity",
        "avg_risk_score",
        "max_risk_score",
        "high_risk_function_count",
        "coverage_ratio",
        "typed_ratio",
        "hotspot_score",
        "static_error_count",
        "tags",
        "owners",
    )

    def list_modules(self) -> list[str]:
        """
        List module identifiers for the repo/commit.

        Returns
        -------
        list[str]
            Sorted module names for the current snapshot.
        """
        tbl = self._ibis_table("core.modules")
        expr = tbl.select("module").order_by("module")

        rows = self._ibis_to_dicts(expr)
        return [str(row["module"]) for row in rows]

    def get_file_summary(self, rel_path: str) -> RowDict | None:
        """
        Return file summary row for a relative path.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        RowDict | None
            File summary row when present.
        """
        tbl = self._ibis_table("docs.v_file_summary")
        expr = tbl.filter(
            and_predicates(
                tbl.rel_path == rel_path,
            )
        ).limit(1)
        summary = self._ibis_to_one(expr)
        if summary is not None:
            return summary

        modules = self._ibis_table("core.modules")
        fallback_expr = modules.filter(
            and_predicates(
                modules.path == rel_path,
            )
        ).limit(1)
        module_row = self._ibis_to_one(fallback_expr)
        if module_row is None:
            return None

        fallback: RowDict = {}
        for key in self._FILE_SUMMARY_COLUMNS:
            fallback[key] = None
        fallback["repo"] = module_row.get("repo")
        fallback["commit"] = module_row.get("commit")
        fallback["rel_path"] = module_row.get("path")
        fallback["module"] = module_row.get("module")
        fallback["language"] = module_row.get("language")
        fallback["tags"] = module_row.get("tags")
        fallback["owners"] = module_row.get("owners")
        return fallback

    def get_module_architecture(self, module: str) -> RowDict | None:
        """
        Return module architecture row.

        Parameters
        ----------
        module
            Module name to look up.

        Returns
        -------
        RowDict | None
            Module architecture when found.
        """
        tbl = self._ibis_table("docs.v_module_architecture")
        expr = tbl.filter(
            and_predicates(
                tbl.module == module,
            )
        ).limit(1)
        return self._ibis_to_one(expr)

    def get_module_profile(self, module: str) -> RowDict | None:
        """
        Return module profile row.

        Parameters
        ----------
        module
            Module name to look up.

        Returns
        -------
        RowDict | None
            Module profile when found.
        """
        tbl = self._ibis_table("analytics.module_profile")
        expr = tbl.filter(
            and_predicates(
                tbl.module == module,
            )
        ).limit(1)
        return self._ibis_to_one(expr, table_key="analytics.module_profile")

    def get_file_profile(self, rel_path: str) -> RowDict | None:
        """
        Return file profile row.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        RowDict | None
            File profile when present.
        """
        tbl = self._ibis_table("analytics.file_profile")
        expr = tbl.filter(
            and_predicates(
                tbl.rel_path == rel_path,
            )
        ).limit(1)
        return self._ibis_to_one(expr, table_key="analytics.file_profile")

    def get_file_hints(self, rel_path: str) -> list[RowDict]:
        """
        Return IDE hints for a given file path.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        list[RowDict]
            Hint rows for the requested file.
        """
        tbl = self._ibis_table("docs.v_ide_hints")
        expr = tbl.filter(
            and_predicates(
                tbl.rel_path == rel_path,
            )
        )
        return self._ibis_to_dicts(expr)
