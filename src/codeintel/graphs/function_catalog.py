"""Shared function metadata catalog for graph builders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from codeintel.graphs.function_index import FunctionSpan, FunctionSpanIndex
from codeintel.ingestion.paths import normalize_rel_path
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map


@dataclass(frozen=True)
class FunctionMeta:
    """Function metadata used across graph builders."""

    goid: int
    urn: str
    rel_path: str
    qualname: str
    start_line: int
    end_line: int


class FunctionCatalog:
    """Centralized access to function spans, URNs, and module mappings."""

    def __init__(
        self,
        *,
        functions: Iterable[FunctionMeta],
        module_by_path: dict[str, str],
    ) -> None:
        self._functions: list[FunctionMeta] = list(functions)
        self._index = FunctionSpanIndex(
            [
                FunctionSpan(
                    goid=fn.goid,
                    rel_path=fn.rel_path,
                    qualname=fn.qualname,
                    start_line=fn.start_line,
                    end_line=fn.end_line,
                )
                for fn in self._functions
            ]
        )
        self._urn_by_goid = {fn.goid: fn.urn for fn in self._functions}
        self._module_by_path = {
            normalize_rel_path(path): mod for path, mod in module_by_path.items()
        }
        self._funcs_by_path: dict[str, list[FunctionMeta]] = {}
        for fn in self._functions:
            self._funcs_by_path.setdefault(fn.rel_path, []).append(fn)

    @property
    def function_spans(self) -> list[FunctionSpan]:
        """All function spans in the catalog."""
        spans: list[FunctionSpan] = []
        for path in self._index.paths():
            spans.extend(self._index.spans_for_path(path))
        return spans

    @property
    def function_index(self) -> FunctionSpanIndex:
        """
        Span lookup index.

        Returns
        -------
        FunctionSpanIndex
            Index supporting path/span lookups.
        """
        return self._index

    @property
    def functions_by_path(self) -> dict[str, list[FunctionMeta]]:
        """
        Functions keyed by normalized path.

        Returns
        -------
        dict[str, list[FunctionMeta]]
            Mapping of relative path to function metadata.
        """
        return self._funcs_by_path

    @property
    def module_by_path(self) -> dict[str, str]:
        """
        Module name mapping keyed by normalized path.

        Returns
        -------
        dict[str, str]
            Mapping of relative path to module name.
        """
        return self._module_by_path

    def urn_for_goid(self, goid: int) -> str | None:
        """
        Return URN for a GOID if known.

        Returns
        -------
        str | None
            URN string when present, otherwise None.
        """
        return self._urn_by_goid.get(goid)

    def lookup_goid(
        self, rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:
        """
        Resolve GOID from span and optional qualname.

        Returns
        -------
        int | None
            GOID value when resolved, otherwise None.
        """
        return self._index.lookup(rel_path, start_line, end_line, qualname)


def load_function_catalog(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> FunctionCatalog:
    """
    Load function metadata and module map for a repo snapshot via a gateway.

    Returns
    -------
    FunctionCatalog
        Catalog containing spans, URNs, and module mapping.
    """
    con = gateway.con
    rows = con.execute(
        """
        SELECT goid_h128, urn, rel_path, qualname, start_line, end_line
        FROM core.goids
        WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
        """,
        [repo, commit],
    ).fetchall()

    functions: list[FunctionMeta] = []
    for goid_h128, urn, rel_path, qualname, start_line, end_line in rows:
        if start_line is None:
            continue
        end_val = int(end_line) if end_line is not None else int(start_line)
        functions.append(
            FunctionMeta(
                goid=int(goid_h128),
                urn=str(urn),
                rel_path=normalize_rel_path(rel_path),
                qualname=str(qualname),
                start_line=int(start_line),
                end_line=end_val,
            )
        )

    module_by_path = load_module_map(gateway, repo, commit)
    return FunctionCatalog(functions=functions, module_by_path=module_by_path)
