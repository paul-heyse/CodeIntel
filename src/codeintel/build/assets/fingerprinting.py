"""Asset fingerprinting helpers.

This module computes deterministic fingerprints for datasets and artifacts
without depending on storage accessors.

Fingerprinting uses content-addressed hashing (STABLE_V1 mode) that excludes
repo/commit, enabling cross-commit asset reuse through upstream version hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from codeintel.core.schemas.hashing import schema_hash

if TYPE_CHECKING:
    from codeintel.core.schemas.provider import SchemaProvider


class FingerprintMode(Enum):
    """Fingerprinting mode for asset version hashes.

    Attributes
    ----------
    STABLE_V1
        Content-addressed mode that excludes repo/commit.
        Uses upstream version hashes for lineage-aware fingerprinting
        that enables cross-commit asset reuse.
    """

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
    """

    table_key: str
    schema_hash: str | None
    row_count: int | None
    upstream_versions: tuple[str, ...]
    options_hash: str | None


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
    """

    artifact_name: str
    artifact_type: str
    size_bytes: int | None
    upstream_versions: tuple[str, ...]
    options_hash: str | None


@dataclass(frozen=True)
class FingerprintPolicy:
    """Policy for computing asset version fingerprints.

    Attributes
    ----------
    mode
        The fingerprinting mode to use (STABLE_V1 for content-addressed).

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
        # Content-addressed fingerprinting based on policy mode
        parts = [
            self.mode.value,
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
        # Content-addressed fingerprinting based on policy mode
        parts = [
            self.mode.value,
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


def compute_table_schema_hash(table_key: str, *, schema_provider: SchemaProvider) -> str | None:
    """Return a deterministic schema hash for a known dataset table_key.

    Parameters
    ----------
    table_key
        Fully-qualified dataset table key (e.g., "analytics.function_metrics").
    schema_provider
        Schema provider used to resolve the table schema.

    Returns
    -------
    str | None
        SHA256 hex digest of (column_name:type) pairs, or None if table_key
        is not registered or has no schema (e.g., view).
    """
    schema = schema_provider.get_table_schema(table_key)
    if schema is None:
        return None
    return schema_hash(schema)


__all__ = [
    "DEFAULT_FINGERPRINT_POLICY",
    "ArtifactVersionInput",
    "FingerprintMode",
    "FingerprintPolicy",
    "TableVersionInput",
    "compute_table_schema_hash",
]
