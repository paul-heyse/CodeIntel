"""Repository for function-centric queries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import ibis.expr.types as it
import pandas as pd

from codeintel.storage.ibis_types import and_predicates, ge, ibis_bool
from codeintel.storage.pandera_schemas import validate_dataset_df
from codeintel.storage.repositories.base import BaseRepository, RowDict


@dataclass(frozen=True)
class FunctionRepository(BaseRepository):
    """Read functions, risk, tests, and architecture details."""

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

    def resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Resolve a function GOID using available identifiers.

        Parameters
        ----------
        urn
            Optional URN to look up.
        goid_h128
            If already known, returns directly.
        rel_path
            File path for lookup when combined with qualname.
        qualname
            Qualified name for lookup when combined with rel_path.

        Returns
        -------
        int | None
            Resolved GOID when found, otherwise ``None``.

        Raises
        ------
        ValueError
            When a GOID value exists but is of an unexpected type.
        """
        if goid_h128 is not None:
            return goid_h128

        goids = self._ibis_table("core.goids")
        expr = goids.filter(and_predicates(goids.repo == self.repo, goids.commit == self.commit))

        if urn:
            expr = expr.filter(ibis_bool(goids.urn == urn))
        elif rel_path and qualname:
            expr = expr.filter(
                and_predicates(goids.rel_path == rel_path, goids.qualname == qualname)
            )
        else:
            # No valid filter criteria
            return None

        records = self._validated_records("core.goids", expr.limit(1))

        if not records:
            return None
        value = records[0].get("goid_h128")
        if value is None:
            return None
        if isinstance(value, (int, float, str, Decimal)):
            return int(value)
        message = f"Unexpected goid type: {type(value)!r}"
        raise ValueError(message)

    def get_function_summary_by_goid(self, goid_h128: int) -> RowDict | None:
        """
        Fetch a function summary row by GOID.

        Parameters
        ----------
        goid_h128
            Function GOID to look up.

        Returns
        -------
        RowDict | None
            Summary row when found, otherwise ``None``.
        """
        table = self._ibis_table("docs.v_function_summary")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.function_goid_h128 == goid_h128,
            )
        ).limit(1)
        records = self._validated_records("docs.v_function_summary", expr)
        return records[0] if records else None

    def list_function_summaries_for_file(self, rel_path: str) -> list[RowDict]:
        """
        List function summaries for a specific file.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        list[RowDict]
            Function summary rows ordered by qualname.
        """
        table = self._ibis_table("docs.v_function_summary")
        expr = table.filter(
            and_predicates(
                table.rel_path == rel_path,
                table.repo == self.repo,
                table.commit == self.commit,
            )
        ).order_by(table.qualname)
        return self._validated_records("docs.v_function_summary", expr)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float,
        limit: int,
        tested_only: bool,
    ) -> list[RowDict]:
        """
        List high-risk functions ordered by risk score.

        Parameters
        ----------
        min_risk
            Minimum risk score threshold.
        limit
            Maximum number of results.
        tested_only
            If True, only include tested functions.

        Returns
        -------
        list[RowDict]
            High-risk function rows limited by ``limit``.
        """
        table = self._ibis_table("analytics.goid_risk_factors")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                ge(table.risk_score, min_risk),
            )
        )
        if tested_only:
            expr = expr.filter(ibis_bool(table.tested == True))  # noqa: E712
        expr = expr.order_by(table.risk_score.desc()).limit(limit)
        return self._validated_records("analytics.goid_risk_factors", expr)

    def get_function_profile(self, goid_h128: int) -> RowDict | None:
        """
        Fetch a function profile by GOID.

        Parameters
        ----------
        goid_h128
            Function GOID to look up.

        Returns
        -------
        RowDict | None
            Function profile row when found.
        """
        table = self._ibis_table("analytics.function_profile")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.function_goid_h128 == goid_h128,
            )
        ).limit(1)
        records = self._validated_records("analytics.function_profile", expr)
        return records[0] if records else None

    def get_function_architecture(self, goid_h128: int) -> RowDict | None:
        """
        Fetch function architecture metrics by GOID.

        Parameters
        ----------
        goid_h128
            Function GOID to look up.

        Returns
        -------
        RowDict | None
            Architecture row when present.
        """
        table = self._ibis_table("docs.v_function_architecture")
        expr = table.filter(
            and_predicates(
                table.repo == self.repo,
                table.commit == self.commit,
                table.function_goid_h128 == goid_h128,
            )
        ).limit(1)
        records = self._ibis_to_dicts(expr)
        return records[0] if records else None

    def list_function_goids(self) -> list[int]:
        """
        Return all function GOIDs for the repo/commit.

        Returns
        -------
        list[int]
            Function GOIDs present in the snapshot.
        """
        table = self._ibis_table("docs.v_function_summary")
        expr = table.filter(
            and_predicates(table.repo == self.repo, table.commit == self.commit)
        ).select("function_goid_h128")

        # NOTE: Skip Pandera validation for single-column projections.
        df = pd.DataFrame(expr.execute())
        records = df.where(pd.notna(df), None).to_dict(orient="records")

        goids: list[int] = []
        for row in records:
            raw = row.get("function_goid_h128")
            if raw is None:
                continue
            goids.append(int(raw))
        return goids
