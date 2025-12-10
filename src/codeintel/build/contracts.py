"""Output contracts for build targets.

This module defines the contract types that specify what an OutputTarget
produces. The OutputContract is the single source of truth for:
- Table schemas (columns, types, constraints)
- File artifacts (SCIP indexes, exports, etc.)

By making contracts authoritative, we can:
- Derive TABLE_SCHEMAS from target definitions
- Validate plugin outputs at write time
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

from collections.abc import Iterable
from dataclasses import dataclass

# Re-export schema primitives for convenience
from codeintel.config.datasets.primitives import (
    Column,
    ColumnType,
    Index,
    TableSchema,
)

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
        These schemas are authoritative - TABLE_SCHEMAS is derived from them.
    artifacts
        Tuple of ArtifactSpec definitions for files this target produces.
        Empty for targets that only write to tables.

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

        # Check for duplicate table keys
        table_keys = [t.fq_name for t in self.tables]
        seen_tables: set[str] = set()
        for key in table_keys:
            if key in seen_tables:
                errors.append(f"Duplicate table key: {key}")
            seen_tables.add(key)

        # Check for duplicate artifact names
        seen_artifacts: set[str] = set()
        for artifact in self.artifacts:
            if artifact.name in seen_artifacts:
                errors.append(f"Duplicate artifact name: {artifact.name}")
            seen_artifacts.add(artifact.name)

        # Check table schemas have at least one column
        errors.extend(
            f"Table {table.fq_name} has no columns" for table in self.tables if not table.columns
        )

        return errors

    @classmethod
    def simple(
        cls,
        table_keys: Iterable[str],
        artifacts: Iterable[ArtifactSpec] = (),
    ) -> OutputContract:
        """Build a lightweight contract from table keys and optional artifacts.

        This helper is intended for tests and quick constructions where only
        fully-qualified table names are needed. It materializes minimal
        TableSchema instances with placeholder columns to satisfy validation.

        Parameters
        ----------
        table_keys
            Iterable of fully-qualified table names (e.g., "core.goids").
        artifacts
            Optional iterable of ArtifactSpec definitions.

        Returns
        -------
        OutputContract
            Contract containing the provided table keys and artifacts.

        Raises
        ------
        ValueError
            If any table key is missing a schema or name component.
        """
        tables: list[TableSchema] = []
        for key in table_keys:
            if "." not in key:
                message = f"Table key must include schema and name: {key}"
                raise ValueError(message)
            schema, name = key.split(".", 1)
            tables.append(
                TableSchema(
                    schema=schema,
                    name=name,
                    columns=[Column("__placeholder", "VARCHAR", nullable=True)],
                    description="Placeholder schema generated by OutputContract.simple",
                )
            )
        return cls(tables=tuple(tables), artifacts=tuple(artifacts))


# Empty contract for targets that don't produce tables
EMPTY_CONTRACT: OutputContract = OutputContract()
