"""Output contracts for build targets.

This module defines the contract types that specify what a TargetDescriptor
produces. The OutputContract is the authoritative list of:
- Table outputs (table keys + optional write policy or schema digest pointer)
- File artifacts (SCIP indexes, exports, etc.)

By making contracts authoritative, we can:
- Validate target outputs at write time
- Track artifacts as first-class outputs

Example
-------
>>> from codeintel.build.contracts import OutputContract, ArtifactSpec, TableOutputDescriptor
>>> contract = OutputContract(
...     tables=(TableOutputDescriptor(table_key="core.symbols"),),
...     artifacts=(ArtifactSpec("scip_index", "{scip_dir}/index.scip"),),
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from codeintel.config.datasets.primitives import Column, ColumnType, Index, TableSchema
from codeintel.core.schemas.primitives import TableWritePolicy

__all__ = [
    "ArtifactSpec",
    "Column",
    "ColumnType",
    "Index",
    "OutputContract",
    "TableOutputDescriptor",
    "TableSchema",
    "TableWritePolicy",
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
class TableOutputDescriptor:
    """Descriptor for a table output produced by a target.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    schema_digest
        Optional schema digest pointer for persisted schema registry entries.
    write_policy
        Optional write policy override for materialization.
    """

    table_key: str
    schema_digest: str | None = None
    write_policy: TableWritePolicy | None = None


@dataclass(frozen=True)
class OutputContract:
    """Contract defining what a target produces.

    The OutputContract is the single source of truth for a target's
    output identities. It specifies database table keys (plus optional
    write policy or schema digest pointers) and file artifacts.

    Parameters
    ----------
    tables
        Tuple of TableOutputDescriptor entries for tables this target writes.
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
    >>> from codeintel.build.contracts import TableOutputDescriptor
    >>> contract = OutputContract(
    ...     tables=(
    ...         TableOutputDescriptor(table_key="core.goids"),
    ...     ),
    ...     artifacts=(ArtifactSpec("scip_index", "{scip_dir}/index.scip"),),
    ... )
    """

    tables: tuple[TableOutputDescriptor, ...] = ()
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
        return tuple(t.table_key for t in self.tables)

    @property
    def artifact_names(self) -> tuple[str, ...]:
        """Return artifact names.

        Returns
        -------
        tuple[str, ...]
            Names of all artifacts in this contract.
        """
        return tuple(a.name for a in self.artifacts)

    def get_table(self, table_key: str) -> TableOutputDescriptor | None:
        """Look up a table descriptor by key.

        Parameters
        ----------
        table_key
            Fully-qualified table name (e.g., "core.goids").

        Returns
        -------
        TableOutputDescriptor | None
            The descriptor if found, None otherwise.
        """
        for table in self.tables:
            if table.table_key == table_key:
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

        table_keys = [t.table_key for t in self.tables]
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
            f"Table output has invalid key: {table.table_key!r}"
            for table in self.tables
            if not table.table_key or "." not in table.table_key
        )

        return errors


EMPTY_CONTRACT: OutputContract = OutputContract()
