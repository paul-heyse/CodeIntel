"""Pure GOID computation functions.

This module provides stateless functions for computing GOIDs and URNs
without any database or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.tabular.kernels import hash_struct_goid
from codeintel.core.columnar.iter import iter_array_values
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.data_models.rows import GoidCrosswalkRow, GoidRow

if TYPE_CHECKING:
    from datetime import datetime

DECIMAL_38_MAX = 10**38 - 1


@dataclass(frozen=True)
class GoidDescriptor:
    """Descriptor for a single code entity.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    language
        Programming language.
    rel_path
        Relative file path.
    kind
        Entity kind (module, class, function, method).
    qualname
        Fully qualified name.
    start_line
        Starting line number.
    end_line
        Optional ending line number.
    """

    repo: str
    commit: str
    language: str
    rel_path: str
    kind: str
    qualname: str
    start_line: int
    end_line: int | None


@dataclass(frozen=True)
class GoidResult:
    """Result of GOID computation.

    Attributes
    ----------
    goid_h128
        128-bit GOID hash as integer.
    urn
        GOID URN string.
    descriptor
        Original descriptor used for computation.
    """

    goid_h128: int
    urn: str
    descriptor: GoidDescriptor


_GOID_HASH_COLUMNS = (
    "repo",
    "commit",
    "language",
    "rel_path",
    "kind",
    "qualname",
    "start_line",
    "end_line",
)


def _goid_hash_table(descriptor: GoidDescriptor) -> pa.Table:
    return pa.table(
        {
            "repo": pa.array([descriptor.repo], type=pa.string()),
            "commit": pa.array([descriptor.commit], type=pa.string()),
            "language": pa.array([descriptor.language], type=pa.string()),
            "rel_path": pa.array([descriptor.rel_path], type=pa.string()),
            "kind": pa.array([descriptor.kind], type=pa.string()),
            "qualname": pa.array([descriptor.qualname], type=pa.string()),
            "start_line": pa.array([descriptor.start_line], type=pa.int64()),
            "end_line": pa.array([descriptor.end_line], type=pa.int64()),
        }
    )


def compute_goid(descriptor: GoidDescriptor) -> int:
    """Compute a stable 128-bit GOID integer from an entity descriptor.

    The GOID is derived from Arrow's hash kernel over the descriptor fields,
    yielding a stable DECIMAL(38,0)-safe identifier.

    Parameters
    ----------
    descriptor
        Metadata describing the code entity.

    Returns
    -------
    int
        Stable 128-bit integer representation of the GOID.

    Raises
    ------
    ValueError
        If the hash value cannot be normalized to a DECIMAL(38,0) integer.

    Examples
    --------
    >>> desc = GoidDescriptor(
    ...     repo="myrepo",
    ...     commit="abc123",
    ...     language="python",
    ...     rel_path="module.py",
    ...     kind="function",
    ...     qualname="module.func",
    ...     start_line=10,
    ...     end_line=20,
    ... )
    >>> goid = compute_goid(desc)
    >>> isinstance(goid, int)
    True
    """
    table = _goid_hash_table(descriptor)
    hashed = hash_struct_goid(table, columns=_GOID_HASH_COLUMNS)
    value = next(iter_array_values(hashed), None)
    normalized = normalize_decimal_id(value)
    if normalized is None:
        msg = "Failed to normalize GOID hash value."
        raise ValueError(msg)
    return normalized


def build_urn(descriptor: GoidDescriptor) -> str:
    """Build a GOID URN from a descriptor.

    The URN format encodes repository, path, kind, and span information
    in a human-readable and parseable format.

    Parameters
    ----------
    descriptor
        Metadata describing the code entity.

    Returns
    -------
    str
        GOID URN encoding all descriptor fields.

    Examples
    --------
    >>> desc = GoidDescriptor(
    ...     repo="myrepo",
    ...     commit="abc123",
    ...     language="python",
    ...     rel_path="module.py",
    ...     kind="function",
    ...     qualname="module.func",
    ...     start_line=10,
    ...     end_line=20,
    ... )
    >>> urn = build_urn(desc)
    >>> urn.startswith("goid:")
    True
    """
    base = (
        f"goid:{descriptor.repo}/{descriptor.rel_path}"
        f"#{descriptor.language}:{descriptor.kind}:{descriptor.qualname}"
    )
    if descriptor.end_line is None:
        return f"{base}?s={descriptor.start_line}"
    return f"{base}?s={descriptor.start_line}&e={descriptor.end_line}"


def compute_goid_result(descriptor: GoidDescriptor) -> GoidResult:
    """Compute both GOID and URN from a descriptor.

    Parameters
    ----------
    descriptor
        Metadata describing the code entity.

    Returns
    -------
    GoidResult
        Combined GOID hash and URN.
    """
    return GoidResult(
        goid_h128=compute_goid(descriptor),
        urn=build_urn(descriptor),
        descriptor=descriptor,
    )


def determine_kind(
    node_type: str,
    parent_qualname: str | None,
    _rel_path: str,
    module_name: str,
) -> str:
    """Determine the entity kind from AST node type and context.

    Parameters
    ----------
    node_type
        AST node type (Module, ClassDef, FunctionDef, AsyncFunctionDef).
    parent_qualname
        Parent's qualified name if any.
    _rel_path
        Relative file path (reserved for future use).
    module_name
        Module name derived from path.

    Returns
    -------
    str
        Entity kind: module, class, function, or method.

    Examples
    --------
    >>> determine_kind("Module", None, "pkg/mod.py", "pkg.mod")
    'module'
    >>> determine_kind("FunctionDef", "MyClass", "mod.py", "mod")
    'method'
    >>> determine_kind("FunctionDef", None, "mod.py", "mod")
    'function'
    """
    if node_type == "Module":
        return "module"
    if node_type == "ClassDef":
        return "class"
    if parent_qualname and parent_qualname != module_name:
        return "method"
    return "function"


def build_goid_row(
    descriptor: GoidDescriptor,
    goid_h128: int,
    urn: str,
    created_at: datetime,
) -> GoidRow:
    """Build a GoidRow from computation results.

    Parameters
    ----------
    descriptor
        Original entity descriptor.
    goid_h128
        Computed GOID hash.
    urn
        Computed URN.
    created_at
        Creation timestamp.

    Returns
    -------
    GoidRow
        Row ready for persistence.
    """
    return GoidRow(
        goid_h128=goid_h128,
        urn=urn,
        repo=descriptor.repo,
        commit=descriptor.commit,
        rel_path=descriptor.rel_path,
        language=descriptor.language,
        kind=descriptor.kind,
        qualname=descriptor.qualname,
        start_line=descriptor.start_line,
        end_line=descriptor.end_line,
        created_at=created_at,
    )


def build_crosswalk_row(
    descriptor: GoidDescriptor,
    urn: str,
    module_path: str,
    updated_at: datetime,
) -> GoidCrosswalkRow:
    """Build a GoidCrosswalkRow from computation results.

    Parameters
    ----------
    descriptor
        Original entity descriptor.
    urn
        Computed URN.
    module_path
        Module path for the entity.
    updated_at
        Update timestamp.

    Returns
    -------
    GoidCrosswalkRow
        Crosswalk row ready for persistence.
    """
    return GoidCrosswalkRow(
        repo=descriptor.repo,
        commit=descriptor.commit,
        goid=urn,
        lang=descriptor.language,
        module_path=module_path,
        file_path=descriptor.rel_path,
        start_line=descriptor.start_line,
        end_line=descriptor.end_line,
        scip_symbol=None,
        ast_qualname=descriptor.qualname,
        cst_node_id=None,
        chunk_id=None,
        symbol_id=None,
        updated_at=updated_at,
    )


__all__ = [
    "DECIMAL_38_MAX",
    "GoidCrosswalkRow",
    "GoidDescriptor",
    "GoidResult",
    "GoidRow",
    "build_crosswalk_row",
    "build_goid_row",
    "build_urn",
    "compute_goid",
    "compute_goid_result",
    "determine_kind",
]
