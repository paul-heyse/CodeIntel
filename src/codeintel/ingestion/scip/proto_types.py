"""Typing protocols for generated SCIP protobuf bindings."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol


class SignatureDocumentationProto(Protocol):
    """Protocol for signature documentation payloads."""

    text: str


class StringListProto(Protocol):
    """Protocol for repeated string fields."""

    def __iter__(self) -> Iterator[str]:
        """Iterate over string values."""
        ...

    def __len__(self) -> int:
        """Return the number of string values."""
        ...

    def __getitem__(self, index: int) -> str:
        """Return a string value by index."""
        ...

    def append(self, value: str) -> None:
        """Append a single string value."""
        ...

    def extend(self, values: Iterable[str]) -> None:
        """Extend with multiple string values."""
        ...


class IntListProto(Protocol):
    """Protocol for repeated integer fields."""

    def __iter__(self) -> Iterator[int]:
        """Iterate over integer values."""
        ...

    def __len__(self) -> int:
        """Return the number of integer values."""
        ...

    def __getitem__(self, index: int) -> int:
        """Return an integer value by index."""
        ...

    def append(self, value: int) -> None:
        """Append a single integer value."""
        ...

    def extend(self, values: Iterable[int]) -> None:
        """Extend with multiple integer values."""
        ...


class RelationshipProto(Protocol):
    """Protocol for symbol relationship entries."""

    symbol: str
    is_reference: bool
    is_implementation: bool
    is_type_definition: bool
    is_definition: bool


class RelationshipListProto(Protocol):
    """Protocol for relationship container lists."""

    def __iter__(self) -> Iterator[RelationshipProto]:
        """Iterate over relationships."""
        ...

    def __len__(self) -> int:
        """Return the number of relationships."""
        ...

    def __getitem__(self, index: int) -> RelationshipProto:
        """Return a relationship by index."""
        ...

    def add(self) -> RelationshipProto:
        """Append a new relationship and return it."""
        ...

    def extend(self, values: Iterable[RelationshipProto]) -> None:
        """Extend with multiple relationships."""
        ...


class SymbolInfoProto(Protocol):
    """Protocol for SCIP symbol information messages."""

    symbol: str
    documentation: StringListProto
    kind: int
    display_name: str
    signature_documentation: SignatureDocumentationProto | None
    enclosing_symbol: str
    relationships: RelationshipListProto


class OccurrenceDiagnosticProto(Protocol):
    """Protocol for diagnostic entries on occurrences."""

    severity: int
    code: str
    message: str
    source: str


class DiagnosticListProto(Protocol):
    """Protocol for diagnostic container lists."""

    def __iter__(self) -> Iterator[OccurrenceDiagnosticProto]:
        """Iterate over diagnostics."""
        ...

    def __len__(self) -> int:
        """Return the number of diagnostics."""
        ...

    def __getitem__(self, index: int) -> OccurrenceDiagnosticProto:
        """Return a diagnostic by index."""
        ...

    def add(self) -> OccurrenceDiagnosticProto:
        """Append a new diagnostic and return it."""
        ...

    def extend(self, values: Iterable[OccurrenceDiagnosticProto]) -> None:
        """Extend with multiple diagnostics."""
        ...


class OccurrenceProto(Protocol):
    """Protocol for SCIP occurrences."""

    symbol: str
    range: IntListProto
    symbol_roles: int
    diagnostics: DiagnosticListProto


class SymbolListProto(Protocol):
    """Protocol for symbol container lists."""

    def __iter__(self) -> Iterator[SymbolInfoProto]:
        """Iterate over symbol information entries."""
        ...

    def __len__(self) -> int:
        """Return the number of symbol entries."""
        ...

    def __getitem__(self, index: int) -> SymbolInfoProto:
        """Return a symbol entry by index."""
        ...

    def add(self) -> SymbolInfoProto:
        """Append a new symbol and return it."""
        ...


class OccurrenceListProto(Protocol):
    """Protocol for occurrence container lists."""

    def __iter__(self) -> Iterator[OccurrenceProto]:
        """Iterate over occurrence entries."""
        ...

    def __len__(self) -> int:
        """Return the number of occurrence entries."""
        ...

    def __getitem__(self, index: int) -> OccurrenceProto:
        """Return an occurrence entry by index."""
        ...

    def add(self) -> OccurrenceProto:
        """Append a new occurrence and return it."""
        ...


class DocumentProto(Protocol):
    """Protocol for SCIP document messages."""

    relative_path: str
    symbols: SymbolListProto
    occurrences: OccurrenceListProto
    position_encoding: int


class DocumentListProto(Protocol):
    """Protocol for document container lists."""

    def __iter__(self) -> Iterator[DocumentProto]:
        """Iterate over document entries."""
        ...

    def __len__(self) -> int:
        """Return the number of document entries."""
        ...

    def __getitem__(self, index: int) -> DocumentProto:
        """Return a document entry by index."""
        ...

    def add(self) -> DocumentProto:
        """Append a new document and return it."""
        ...


class ExternalSymbolProto(Protocol):
    """Protocol for external symbol messages."""

    symbol: str


class ExternalSymbolListProto(Protocol):
    """Protocol for external symbol container lists."""

    def __iter__(self) -> Iterator[ExternalSymbolProto]:
        """Iterate over external symbol entries."""
        ...

    def __len__(self) -> int:
        """Return the number of external symbol entries."""
        ...

    def __getitem__(self, index: int) -> ExternalSymbolProto:
        """Return an external symbol entry by index."""
        ...

    def add(self) -> ExternalSymbolProto:
        """Append a new external symbol and return it."""
        ...


class IndexProto(Protocol):
    """Protocol for the root SCIP index message."""

    documents: DocumentListProto
    external_symbols: ExternalSymbolListProto
    metadata: MetadataProto


class MetadataProto(Protocol):
    """Protocol for SCIP metadata messages."""

    text_document_encoding: int


class IndexFactory(Protocol):
    """Protocol for a callable that creates IndexProto instances."""

    def __call__(self) -> IndexProto:
        """Instantiate a new IndexProto."""
        ...


class DocumentFactory(Protocol):
    """Protocol for a callable that creates DocumentProto instances."""

    def __call__(self) -> DocumentProto:
        """Instantiate a new DocumentProto."""
        ...


class MetadataFactory(Protocol):
    """Protocol for a callable that creates MetadataProto instances."""

    def __call__(self) -> MetadataProto:
        """Instantiate a new MetadataProto."""
        ...


class SymbolInfoFactory(Protocol):
    """Protocol for a callable that creates SymbolInfoProto instances."""

    def __call__(self) -> SymbolInfoProto:
        """Instantiate a new SymbolInfoProto."""
        ...


class ScipProtoModule(Protocol):
    """Protocol for generated scip_pb2 modules."""

    Index: IndexFactory
    Metadata: MetadataFactory
    Document: DocumentFactory
    SymbolInformation: SymbolInfoFactory
    Severity: object


__all__ = [
    "DiagnosticListProto",
    "DocumentFactory",
    "DocumentListProto",
    "DocumentProto",
    "ExternalSymbolListProto",
    "ExternalSymbolProto",
    "IndexFactory",
    "IndexProto",
    "IntListProto",
    "MetadataFactory",
    "MetadataProto",
    "OccurrenceDiagnosticProto",
    "OccurrenceListProto",
    "OccurrenceProto",
    "RelationshipListProto",
    "RelationshipProto",
    "ScipProtoModule",
    "SignatureDocumentationProto",
    "StringListProto",
    "SymbolInfoFactory",
    "SymbolInfoProto",
    "SymbolListProto",
]
