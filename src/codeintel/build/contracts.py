"""Output contracts for build targets.

This module defines the contract types that specify what an OutputTarget
produces. The OutputContract is the single source of truth for:
- Table schemas (columns, types, constraints)
- File artifacts (SCIP indexes, exports, etc.)

By making contracts authoritative, we can:
- Derive TABLE_SCHEMAS from target definitions
- Validate target outputs at write time
- Track artifacts as first-class outputs

Example
-------
>>> from codeintel.build.contracts import OutputContract, ArtifactSpec
>>> from codeintel.config.datasets.primitives import TableSchema, Column
>>> contract = OutputContract(
...     tables=(TableSchema("core", "symbols", [Column("name", "VARCHAR")]),),
...     artifacts=(ArtifactSpec("scip_index", "{scip_dir}/index.scip"),),
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from codeintel.config.datasets.primitives import Column, ColumnType, Index, TableSchema

__all__ = [
    "ArtifactSpec",
    "Column",
    "ColumnType",
    "Index",
    "OutputContract",
    "TableSchema",
]


@dataclass(frozen=True)
class ArtifactSpec:
    """Specification for a file artifact produced by a target.

    Artifacts are non-table outputs like SCIP indexes, export files,
    or generated documentation. They are tracked separately from
    database tables but are first-class outputs of the build system.

    Parameters
    ----------
    name
        Unique artifact identifier within the target (e.g., "scip_index").
    path_template
        Path template with placeholders for resolution.
        Supported placeholders:
        - {build_dir}: Build output directory
        - {scip_dir}: SCIP artifacts directory
        - {export_dir}: Document export directory
        - {repo_root}: Repository root path
    description
        Optional description of the artifact's purpose.
    required
        Whether this artifact must be produced for target success.
        Defaults to True.

    Examples
    --------
    >>> spec = ArtifactSpec(
    ...     name="scip_index",
    ...     path_template="{scip_dir}/index.scip",
    ...     description="SCIP index file for symbol resolution",
    ... )
    """

    name: str
    path_template: str
    description: str | None = None
    required: bool = True


@dataclass(frozen=True)
class OutputContract:
    """Contract defining what an OutputTarget produces.

    The OutputContract is the single source of truth for a target's
    outputs. It specifies both database tables (with full schemas)
    and file artifacts.

    Parameters
    ----------
    tables
        Tuple of TableSchema definitions for tables this target writes.
        These schemas (and the derived ``table_keys``) are authoritative for
        the target; they must be populated from the canonical table registry.
    artifacts
        Tuple of ArtifactSpec definitions for files this target produces.
        Empty for targets that only write to tables.
    json_schema_ids
        JSON Schema identifiers for export validation.
    jsonl_filenames
        Default JSONL export filenames for tables in this contract.
    parquet_filenames
        Default Parquet export filenames for tables in this contract.
    owner
        Team or individual owner of this target's outputs.
    description
        Human-readable description of this target's purpose.
    family
        Dataset family classification (e.g., "analytics", "core").
    freshness_sla
        Expected freshness guarantee (e.g., "daily", "hourly").
    retention_policy
        Data retention policy descriptor (e.g., "90d").
    upstream_dependencies
        Names of upstream datasets this target depends on.
    tags
        Classification tags for the target outputs.
    validation_profile
        Validation strictness level for outputs.

    Examples
    --------
    >>> from codeintel.config.datasets.primitives import TableSchema, Column
    >>> contract = OutputContract(
    ...     tables=(
    ...         TableSchema(
    ...             schema="core",
    ...             name="goids",
    ...             columns=[
    ...                 Column("goid_h128", "DECIMAL(38,0)", nullable=False),
    ...                 Column("urn", "VARCHAR", nullable=False),
    ...             ],
    ...             primary_key=("goid_h128",),
    ...         ),
    ...     ),
    ...     artifacts=(ArtifactSpec("scip_index", "{scip_dir}/index.scip"),),
    ... )
    """

    tables: tuple[TableSchema, ...] = ()
    artifacts: tuple[ArtifactSpec, ...] = ()

    # Extended metadata for dataset contract derivation (PR-68)
    json_schema_ids: tuple[str, ...] = ()
    jsonl_filenames: tuple[str, ...] = ()
    parquet_filenames: tuple[str, ...] = ()
    owner: str | None = None
    description: str | None = None
    family: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    validation_profile: Literal["strict", "lenient"] = "strict"

    @property
    def table_keys(self) -> tuple[str, ...]:
        """Return fully-qualified table names.

        Returns
        -------
        tuple[str, ...]
            Table keys in "schema.name" format.
        """
        return tuple(t.fq_name for t in self.tables)

    @property
    def artifact_names(self) -> tuple[str, ...]:
        """Return artifact names.

        Returns
        -------
        tuple[str, ...]
            Names of all artifacts in this contract.
        """
        return tuple(a.name for a in self.artifacts)

    def get_table(self, table_key: str) -> TableSchema | None:
        """Look up a table schema by key.

        Parameters
        ----------
        table_key
            Fully-qualified table name (e.g., "core.goids").

        Returns
        -------
        TableSchema | None
            The schema if found, None otherwise.
        """
        for table in self.tables:
            if table.fq_name == table_key:
                return table
        return None

    def get_artifact(self, name: str) -> ArtifactSpec | None:
        """Look up an artifact spec by name.

        Parameters
        ----------
        name
            Artifact identifier.

        Returns
        -------
        ArtifactSpec | None
            The spec if found, None otherwise.
        """
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None

    def validate(self) -> list[str]:
        """Validate the contract for internal consistency.

        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        table_keys = [t.fq_name for t in self.tables]
        seen_tables: set[str] = set()
        for key in table_keys:
            if key in seen_tables:
                errors.append(f"Duplicate table key: {key}")
            seen_tables.add(key)

        seen_artifacts: set[str] = set()
        for artifact in self.artifacts:
            if artifact.name in seen_artifacts:
                errors.append(f"Duplicate artifact name: {artifact.name}")
            seen_artifacts.add(artifact.name)

        errors.extend(
            f"Table {table.fq_name} has no columns" for table in self.tables if not table.columns
        )

        return errors


EMPTY_CONTRACT: OutputContract = OutputContract()
