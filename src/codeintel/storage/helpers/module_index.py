"""Read-side helpers for module metadata from core.modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from codeintel.core.paths import normalize_path
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

LOG: Final = logging.getLogger(__name__)


def load_module_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    language: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """
    Load path->module mapping from core.modules.

    Parameters
    ----------
    gateway :
        Storage gateway bound to the target DuckDB database.
    repo : str
        Repository slug.
    commit : str
        Commit SHA anchoring the snapshot.
    language : str | None, optional
        Optional language filter.
    logger : logging.Logger | None, optional
        Logger for warnings; defaults to module logger.

    Returns
    -------
    dict[str, str]
        Normalized mapping of relative path -> module name.
    """
    relation = gateway.relation_from_table_key("core.modules")
    predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit)
    )
    if language is not None:
        predicate &= ColumnExpression("language") == ConstantExpression(language)
    rows = relation.filter(predicate).select("path", "module").fetchall()
    module_map = {normalize_path(str(path)): str(module) for path, module in rows}
    if not module_map:
        (logger or LOG).warning("No modules found in core.modules for %s@%s", repo, commit)
    return module_map


__all__ = ["load_module_map"]
