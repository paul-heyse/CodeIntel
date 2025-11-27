"""Read-side helpers for module metadata from core.modules."""

from __future__ import annotations

import logging
from typing import Final

from codeintel.ingestion.paths import normalize_rel_path
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
    con = gateway.con
    params: list[object] = [repo, commit]
    query = """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """
    if language is not None:
        query += " AND language = ?"
        params.append(language)
    rows = con.execute(query, params).fetchall()
    module_map = {normalize_rel_path(str(path)): str(module) for path, module in rows}
    if not module_map:
        (logger or LOG).warning("No modules found in core.modules for %s@%s", repo, commit)
    return module_map


__all__ = ["load_module_map"]
