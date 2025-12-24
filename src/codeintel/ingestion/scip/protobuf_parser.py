"""Helpers for parsing SCIP protobuf indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.ports.tools import ScipDocument, ScipOccurrence, ScipSymbol
from codeintel.ingestion.scip.models import (
    ScipDiagnostic,
    ScipExternalSymbol,
    ScipSymbolInfo,
    ScipSymbolRelationship,
)
from codeintel.ingestion.scip.proto import load_generated_module

if TYPE_CHECKING:
    from types import ModuleType


@dataclass(frozen=True)
class ScipParsedIndex:
    """Parsed SCIP index payload."""

    documents: tuple[ScipDocument, ...]
    symbol_infos: tuple[ScipSymbolInfo, ...]
    relationships: tuple[ScipSymbolRelationship, ...]
    diagnostics: tuple[ScipDiagnostic, ...]
    external_symbols: tuple[ScipExternalSymbol, ...]


def load_proto_module(proto_module_path: Path) -> ModuleType:
    """Load scip_pb2 from the generated file path.

    Returns
    -------
    ModuleType
        Imported protobuf module.
    """
    return load_generated_module(proto_module_path)


def load_index(index_path: Path, proto_module_path: Path) -> object:
    """Load a SCIP index using generated protobuf bindings.

    Returns
    -------
    object
        Parsed protobuf Index instance.
    """
    module = load_proto_module(proto_module_path)
    index = module.Index()
    index.ParseFromString(index_path.read_bytes())
    return index


def parse_index(index_path: Path, proto_module_path: Path) -> ScipParsedIndex:
    """Parse index.scip into typed structures.

    Returns
    -------
    ScipParsedIndex
        Parsed index with documents, symbols, and diagnostics.
    """
    module = load_proto_module(proto_module_path)
    index = module.Index()
    index.ParseFromString(index_path.read_bytes())

    documents = tuple(_parse_document(doc) for doc in index.documents)
    symbol_infos: list[ScipSymbolInfo] = []
    relationships: list[ScipSymbolRelationship] = []
    diagnostics: list[ScipDiagnostic] = []

    for doc in index.documents:
        rel_path = doc.relative_path
        for sym_info in doc.symbols:
            symbol_infos.append(_parse_symbol_info(module, sym_info))
            relationships.extend(_parse_relationships(sym_info))
        diagnostics.extend(_parse_document_diagnostics(module, doc, rel_path))

    external_symbols = tuple(_parse_external_symbol(module, sym) for sym in index.external_symbols)

    return ScipParsedIndex(
        documents=documents,
        symbol_infos=tuple(symbol_infos),
        relationships=tuple(relationships),
        diagnostics=tuple(diagnostics),
        external_symbols=external_symbols,
    )


def _parse_document(doc: object) -> ScipDocument:
    symbols = tuple(_parse_symbol(sym) for sym in doc.symbols)
    occurrences = tuple(_parse_occurrence(occ) for occ in doc.occurrences)
    return ScipDocument(
        relative_path=doc.relative_path,
        symbols=symbols,
        occurrences=occurrences,
    )


def _parse_symbol(sym: object) -> ScipSymbol:
    documentation = "\n".join(sym.documentation) if sym.documentation else None
    return ScipSymbol(symbol=sym.symbol, documentation=documentation)


def _parse_occurrence(occ: object) -> ScipOccurrence:
    range_tuple = _parse_range(occ.range)
    if range_tuple is None:
        range_tuple = (0, 0, 0, 0)
    return ScipOccurrence(
        symbol=occ.symbol,
        range_start_line=range_tuple[0],
        range_start_col=range_tuple[1],
        range_end_line=range_tuple[2],
        range_end_col=range_tuple[3],
        symbol_roles=occ.symbol_roles,
    )


def _parse_range(rng: object) -> tuple[int, int, int, int] | None:
    if not isinstance(rng, list):
        return None
    if len(rng) == 3:
        return (int(rng[0]), int(rng[1]), int(rng[0]), int(rng[2]))
    if len(rng) == 4:
        return (int(rng[0]), int(rng[1]), int(rng[2]), int(rng[3]))
    return None


def _parse_symbol_info(module: ModuleType, sym: object) -> ScipSymbolInfo:
    documentation = "\n".join(sym.documentation) if sym.documentation else None
    signature = _signature_text(sym)
    return ScipSymbolInfo(
        symbol=sym.symbol,
        documentation=documentation,
        kind=int(sym.kind) if sym.kind is not None else None,
        display_name=sym.display_name or None,
        signature=signature,
        enclosing_symbol=sym.enclosing_symbol or None,
    )


def _signature_text(sym: object) -> str | None:
    sig_doc = sym.signature_documentation
    if sig_doc is None:
        return None
    text = sig_doc.text if hasattr(sig_doc, "text") else ""
    return text or None


def _parse_relationships(sym: object) -> list[ScipSymbolRelationship]:
    relationships: list[ScipSymbolRelationship] = []
    for rel in sym.relationships:
        if rel.is_reference:
            relationships.append(
                ScipSymbolRelationship(
                    symbol=sym.symbol,
                    related_symbol=rel.symbol,
                    relationship_kind="reference",
                )
            )
        if rel.is_implementation:
            relationships.append(
                ScipSymbolRelationship(
                    symbol=sym.symbol,
                    related_symbol=rel.symbol,
                    relationship_kind="implementation",
                )
            )
        if rel.is_type_definition:
            relationships.append(
                ScipSymbolRelationship(
                    symbol=sym.symbol,
                    related_symbol=rel.symbol,
                    relationship_kind="type_definition",
                )
            )
        if rel.is_definition:
            relationships.append(
                ScipSymbolRelationship(
                    symbol=sym.symbol,
                    related_symbol=rel.symbol,
                    relationship_kind="definition",
                )
            )
    return relationships


def _parse_document_diagnostics(
    module: ModuleType,
    doc: object,
    rel_path: str,
) -> list[ScipDiagnostic]:
    diagnostics: list[ScipDiagnostic] = []
    for occ in doc.occurrences:
        range_tuple = _parse_range(occ.range)
        if range_tuple is None:
            continue
        for diag in occ.diagnostics:
            severity = _severity_name(module, diag.severity)
            diagnostics.append(
                ScipDiagnostic(
                    rel_path=rel_path,
                    start_line=range_tuple[0],
                    start_col=range_tuple[1],
                    end_line=range_tuple[2],
                    end_col=range_tuple[3],
                    severity=severity,
                    code=diag.code or None,
                    message=diag.message,
                    source=diag.source or None,
                )
            )
    return diagnostics


def _severity_name(module: ModuleType, value: int) -> str:
    try:
        return module.Severity.Name(value)
    except Exception:
        return "Unspecified"


def _parse_external_symbol(module: ModuleType, sym: object) -> ScipExternalSymbol:
    manager, name, version = _parse_package_triple(sym.symbol)
    return ScipExternalSymbol(
        symbol=sym.symbol,
        package_manager=manager,
        package_name=name,
        package_version=version,
    )


def _parse_package_triple(symbol: str) -> tuple[str | None, str | None, str | None]:
    if symbol.startswith("local "):
        return None, None, None
    tokens = _split_tokens_with_double_space_escape(symbol)
    if len(tokens) < 4:
        return None, None, None
    manager, name, version = tokens[1], tokens[2], tokens[3]
    return (
        None if manager == "." else manager,
        None if name == "." else name,
        None if version == "." else version,
    )


def _split_tokens_with_double_space_escape(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == " ":
            if idx + 1 < len(text) and text[idx + 1] == " ":
                current.append(" ")
                idx += 2
                continue
            tokens.append("".join(current))
            current = []
            idx += 1
            continue
        current.append(ch)
        idx += 1
    tokens.append("".join(current))
    return tokens


__all__ = [
    "ScipParsedIndex",
    "load_index",
    "load_proto_module",
    "parse_index",
]
