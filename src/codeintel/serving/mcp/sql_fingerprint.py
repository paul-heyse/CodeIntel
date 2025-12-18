"""SQL fingerprint helpers for correlation and sampling."""

from __future__ import annotations

import hashlib

from sqlglot import parse_one
from sqlglot.errors import ParseError


def sqlglot_canonical_sha256(sql: str) -> str:
    """Canonicalize SQL via SQLGlot and return sha256 hex digest.

    Parameters
    ----------
    sql
        SQL string to canonicalize and hash.

    Returns
    -------
    str
        SHA256 hex digest of the canonical SQL form.
    """
    canonical = sql
    try:
        canonical = parse_one(sql, read="duckdb").sql(dialect="duckdb")
    except (ParseError, ValueError):
        canonical = sql
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["sqlglot_canonical_sha256"]
