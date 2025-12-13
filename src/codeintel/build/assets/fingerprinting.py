"""Asset fingerprinting helpers for Phase 4.

This module intentionally lives in the build layer: it computes deterministic
fingerprints for datasets and artifacts without depending on storage accessors.

Fingerprint Policies
--------------------
- FAST: Legacy behavior, includes input_hash which contains repo+commit.
  Fast for skip checks but not suitable for cross-commit reuse.
- STABLE_V1: Content-addressed fingerprint excluding repo+commit.
  Uses upstream version hashes instead of input_hash for lineage-aware
  fingerprinting that enables cross-commit asset reuse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from codeintel.config.datasets.schemas import TABLE_SCHEMAS


class FingerprintMode(Enum):
    """Fingerprinting mode for asset version hashes.

    Attributes
    ----------
    FAST
        Legacy mode that includes input_hash (commit-dependent).
        Suitable for skip checking within a single commit but not
        for cross-commit reuse.
    STABLE_V1
        Content-addressed mode that excludes repo/commit.
        Uses upstream version hashes for lineage-aware fingerprinting
        that enables cross-commit asset reuse.
    """

    FAST = "fast"
    STABLE_V1 = "stable_v1"


@dataclass(frozen=True)
class TableVersionInput:
    """Input parameters for computing a table asset version hash.

    Attributes
    ----------
    table_key
        Fully-qualified table key (e.g., "analytics.function_metrics").
    schema_hash
        Hash of the table schema, or None if not available.
    row_count
        Number of rows in the table, or None if not known.
    upstream_versions
        Version hashes of upstream dependencies.
    options_hash
        Hash of plugin options, or None if no options.
    input_hash
        Legacy input hash (only used in FAST mode).
    """

    table_key: str
    schema_hash: str | None
    row_count: int | None
    upstream_versions: tuple[str, ...]
    options_hash: str | None
    input_hash: str | None = None


@dataclass(frozen=True)
class ArtifactVersionInput:
    """Input parameters for computing an artifact asset version hash.

    Attributes
    ----------
    artifact_name
        Name of the artifact (e.g., "index.scip").
    artifact_type
        Type of the artifact (e.g., "scip", "parquet").
    size_bytes
        Size of the artifact in bytes, or None if not known.
    upstream_versions
        Version hashes of upstream dependencies.
    options_hash
        Hash of plugin options, or None if no options.
    input_hash
        Legacy input hash (only used in FAST mode).
    """

    artifact_name: str
    artifact_type: str
    size_bytes: int | None
    upstream_versions: tuple[str, ...]
    options_hash: str | None
    input_hash: str | None = None


@dataclass(frozen=True)
class FingerprintPolicy:
    """Policy for computing asset version fingerprints.

    This abstraction allows selecting between fast (legacy) fingerprinting
    and stable (cross-commit-capable) fingerprinting.

    Attributes
    ----------
    mode
        The fingerprinting mode to use.

    Examples
    --------
    >>> policy = FingerprintPolicy(mode=FingerprintMode.STABLE_V1)
    >>> inp = TableVersionInput(
    ...     table_key="analytics.function_metrics",
    ...     schema_hash="abc123",
    ...     row_count=100,
    ...     upstream_versions=("v1", "v2"),
    ...     options_hash="def456",
    ... )
    >>> version = policy.compute_table_version_from_input(inp)
    """

    mode: FingerprintMode = FingerprintMode.STABLE_V1

    def compute_table_version(self, inp: TableVersionInput) -> str:
        """Compute version hash for a table asset.

        Parameters
        ----------
        inp
            Table version input containing all required parameters.

        Returns
        -------
        str
            Version hash (16 characters).
        """
        return self.compute_table_version_from_input(inp)

    def compute_table_version_from_input(self, inp: TableVersionInput) -> str:
        """Compute version hash for a table asset from input object.

        Parameters
        ----------
        inp
            Table version input parameters.

        Returns
        -------
        str
            Version hash (16 characters).
        """
        if self.mode == FingerprintMode.FAST:
            return compute_fast_version_hash(
                "table",
                inp.table_key,
                inp.schema_hash,
                inp.row_count,
                inp.input_hash,
                inp.options_hash,
            )

        # STABLE_V1: Content-addressed, commit-independent
        parts = [
            "stable_v1",
            "table",
            inp.table_key,
            inp.schema_hash or "",
            str(inp.row_count or 0),
            inp.options_hash or "",
            *sorted(inp.upstream_versions),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def compute_artifact_version(self, inp: ArtifactVersionInput) -> str:
        """Compute version hash for an artifact asset.

        Parameters
        ----------
        inp
            Artifact version input containing all required parameters.

        Returns
        -------
        str
            Version hash (16 characters).
        """
        return self.compute_artifact_version_from_input(inp)

    def compute_artifact_version_from_input(self, inp: ArtifactVersionInput) -> str:
        """Compute version hash for an artifact asset from input object.

        Parameters
        ----------
        inp
            Artifact version input parameters.

        Returns
        -------
        str
            Version hash (16 characters).
        """
        if self.mode == FingerprintMode.FAST:
            return compute_fast_version_hash(
                "artifact",
                inp.artifact_name,
                inp.artifact_type,
                inp.size_bytes,
                inp.input_hash,
                inp.options_hash,
            )

        # STABLE_V1: Content-addressed, commit-independent
        parts = [
            "stable_v1",
            "artifact",
            inp.artifact_name,
            inp.artifact_type,
            str(inp.size_bytes or 0),
            inp.options_hash or "",
            *sorted(inp.upstream_versions),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


# Default policy for new builds
DEFAULT_FINGERPRINT_POLICY = FingerprintPolicy(mode=FingerprintMode.STABLE_V1)


def _canonical_type(type_str: str) -> str:
    upper = type_str.upper()
    if upper in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMPTZ"
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


def compute_table_schema_hash(table_key: str) -> str | None:
    """Return a deterministic schema hash for a known dataset table_key.

    Parameters
    ----------
    table_key
        Fully-qualified dataset table key (e.g., "analytics.function_metrics").

    Returns
    -------
    str | None
        SHA256 hex digest of (column_name:type) pairs, or None if table_key
        is not registered or has no schema (e.g., view).
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        return None
    if schema.columns is None:
        return None
    parts = [f"{column.name}:{_canonical_type(column.type)}" for column in schema.columns]
    normalized = "|".join(parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_fast_version_hash(*parts: object) -> str:
    """Compute a fast, stable version hash from a tuple of stable components.

    Notes
    -----
    This intentionally produces a short hash for ergonomics. It is content
    addressed but not collision-proof; Phase 4 can upgrade to a stronger policy.

    Returns
    -------
    str
        Hex digest truncated to 16 characters.
    """
    normalized = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "DEFAULT_FINGERPRINT_POLICY",
    "ArtifactVersionInput",
    "FingerprintMode",
    "FingerprintPolicy",
    "TableVersionInput",
    "compute_fast_version_hash",
    "compute_table_schema_hash",
]
