"""Repository for subsystem-related queries."""

from __future__ import annotations

from dataclasses import dataclass

import ibis.expr.types as it
import pandas as pd

from codeintel.storage.ibis_types import and_predicates, count_gt, ilike, or_predicates
from codeintel.storage.pandera_schemas import validate_dataset_df
from codeintel.storage.repositories.base import BaseRepository, RowDict


@dataclass(frozen=True)
class SubsystemRepository(BaseRepository):
    """Read subsystem summaries and memberships."""

    @staticmethod
    def _validated_records(table_key: str, expr: it.Table) -> list[RowDict]:
        """
        Execute an Ibis expression and return validated row dictionaries.

        Parameters
        ----------
        table_key
            Dataset key used for Pandera validation.
        expr
            Ibis table expression to execute.

        Returns
        -------
        list[RowDict]
            Validated records with ``None`` substituted for missing values.
        """
        df = pd.DataFrame(expr.execute())
        validated = validate_dataset_df(table_key, df)
        return validated.where(pd.notna(validated), None).to_dict(orient="records")

    def list_subsystems(
        self,
        *,
        limit: int,
        role: str | None = None,
        query: str | None = None,
    ) -> list[RowDict]:
        """
        List subsystem summaries with optional role and search filters.

        Parameters
        ----------
        limit
            Maximum number of results.
        role
            Optional role filter.
        query
            Optional search query for name/description.

        Returns
        -------
        list[RowDict]
            Subsystem summary rows ordered by module count.
        """
        table = self._ibis_table("docs.v_subsystem_summary")
        expr = table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))

        if role:
            modules = self._ibis_table("analytics.subsystem_modules")
            exists_expr = modules.filter(
                and_predicates(
                    modules.repo == self.repo,
                    modules.commit == self.commit,
                    modules.subsystem_id == table.subsystem_id,
                    modules.role == role,
                )
            ).limit(1)
            expr = expr.filter(count_gt(exists_expr.count(), 0))

        if query:
            pattern = f"%{query}%"
            expr = expr.filter(
                or_predicates(ilike(table.name, pattern), ilike(table.description, pattern))
            )

        expr = expr.order_by([table.module_count.desc(), table.subsystem_id]).limit(limit)
        return self._validated_records("docs.v_subsystem_summary", expr)

    def get_subsystem_summary(self, subsystem_id: str) -> RowDict | None:
        """
        Return a single subsystem summary by identifier.

        Parameters
        ----------
        subsystem_id
            The subsystem identifier.

        Returns
        -------
        RowDict | None
            Subsystem summary row when present.
        """
        table = self._ibis_table("docs.v_subsystem_summary")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.subsystem_id == subsystem_id,
            )
        ).limit(1)
        rows = self._validated_records("docs.v_subsystem_summary", expr)
        return rows[0] if rows else None

    def search_subsystems(
        self,
        *,
        limit: int,
        role: str | None = None,
        query: str | None = None,
    ) -> list[RowDict]:
        """
        Alias for list_subsystems to make intent explicit.

        Parameters
        ----------
        limit
            Maximum number of results.
        role
            Optional role filter.
        query
            Optional search query.

        Returns
        -------
        list[RowDict]
            Subsystem rows matching the search parameters.
        """
        return self.list_subsystems(limit=limit, role=role, query=query)

    def list_subsystem_modules(self, subsystem_id: str) -> list[RowDict]:
        """
        Return module memberships for a subsystem.

        Parameters
        ----------
        subsystem_id
            The subsystem identifier.

        Returns
        -------
        list[RowDict]
            Module membership rows ordered by module.
        """
        table = self._ibis_table("docs.v_module_with_subsystem")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.subsystem_id == subsystem_id,
            )
        ).order_by(table.module)
        return self._validated_records("docs.v_module_with_subsystem", expr)

    def list_subsystem_memberships(self) -> list[RowDict]:
        """
        Return all subsystem-module memberships for the repo/commit.

        Returns
        -------
        list[RowDict]
            Membership rows keyed by subsystem and module.
        """
        table = self._ibis_table("analytics.subsystem_modules")
        expr = table.filter(
            and_predicates(table.repo == self.repo, table.commit == self.commit)
        ).select("subsystem_id", "module")
        return self._ibis_to_dicts(expr)

    def list_subsystems_for_module(self, module: str) -> list[RowDict]:
        """
        Return subsystem memberships for a module.

        Parameters
        ----------
        module
            The module name.

        Returns
        -------
        list[RowDict]
            Subsystem membership rows for the module.
        """
        table = self._ibis_table("docs.v_module_with_subsystem")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.module == module,
            )
        )
        return self._validated_records("docs.v_module_with_subsystem", expr)

    def _has_cache(self, cache_table: str) -> bool:
        """
        Check if a cache table has data for this repo/commit.

        Parameters
        ----------
        cache_table
            Fully qualified table name for the cache.

        Returns
        -------
        bool
            True if cache has at least one row for this repo/commit.
        """
        table = self._ibis_table(cache_table)
        expr = table.filter(
            and_predicates(table.repo == self.repo, table.commit == self.commit)
        ).limit(1)
        return self._ibis_exists(expr)

    def list_subsystem_profiles(self, *, limit: int) -> list[RowDict]:
        """
        Return subsystem profile rows from docs views.

        Parameters
        ----------
        limit
            Maximum number of results.

        Returns
        -------
        list[RowDict]
            Profile rows ordered by module count then subsystem_id.
        """
        cache_table = "analytics.subsystem_profile_cache"
        if self._has_cache(cache_table):
            table = self._ibis_table(cache_table)
            expr = (
                table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
                .order_by([table.module_count.desc(), table.subsystem_id])
                .limit(limit)
            )
            return self._validated_records(cache_table, expr)

        table = self._ibis_table("docs.v_subsystem_profile")
        expr = (
            table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
            .order_by([table.module_count.desc(), table.subsystem_id])
            .limit(limit)
        )
        return self._validated_records("docs.v_subsystem_profile", expr)

    def list_subsystem_coverage(self, *, limit: int) -> list[RowDict]:
        """
        Return subsystem coverage rollups from docs views.

        Parameters
        ----------
        limit
            Maximum number of results.

        Returns
        -------
        list[RowDict]
            Coverage rows ordered by test count then subsystem_id.
        """
        cache_table = "analytics.subsystem_coverage_cache"
        if self._has_cache(cache_table):
            table = self._ibis_table(cache_table)
            expr = (
                table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
                .order_by(
                    [
                        table.test_count.desc(nulls_first=False),
                        table.subsystem_id,
                    ]
                )
                .limit(limit)
            )
            return self._validated_records(cache_table, expr)

        table = self._ibis_table("docs.v_subsystem_coverage")
        expr = (
            table.filter(and_predicates(table.repo == self.repo, table.commit == self.commit))
            .order_by(
                [
                    table.test_count.desc(nulls_first=False),
                    table.subsystem_id,
                ]
            )
            .limit(limit)
        )
        return self._validated_records("docs.v_subsystem_coverage", expr)
