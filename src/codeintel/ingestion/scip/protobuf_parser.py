"""Helpers for parsing SCIP protobuf indexes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import unquote, urlparse

from codeintel.ingestion.ports.tools import ScipDocument, ScipOccurrence, ScipSymbol
from codeintel.ingestion.scip.models import (
    ScipDiagnostic,
    ScipExternalSymbol,
    ScipSymbolInfo,
    ScipSymbolRelationship,
)
from codeintel.ingestion.scip.proto import load_generated_module
from codeintel.ingestion.scip.proto_types import (
    DocumentProto,
    ExternalSymbolProto,
    IndexProto,
    IntListProto,
    OccurrenceProto,
    ScipProtoModule,
    SymbolInfoProto,
)


@dataclass(frozen=True)
class ScipParsedIndex:
    """Parsed SCIP index payload."""

    documents: tuple[ScipDocument, ...]
    symbol_infos: tuple[ScipSymbolInfo, ...]
    relationships: tuple[ScipSymbolRelationship, ...]
    diagnostics: tuple[ScipDiagnostic, ...]
    external_symbols: tuple[ScipExternalSymbol, ...]
    project_root: Path | None


_RANGE_LEN_SAME_LINE = 3
_RANGE_LEN_FULL = 4
_PACKAGE_TOKEN_COUNT = 4
_WIRE_TYPE_VARINT = 0
_WIRE_TYPE_64BIT = 1
_WIRE_TYPE_LENGTH = 2
_WIRE_TYPE_32BIT = 5
_MAX_VARINT_SHIFT = 64
_TEXT_DOCUMENT_ENCODING_UTF8 = 1
_TEXT_DOCUMENT_ENCODING_UTF16 = 2


@dataclass(frozen=True)
class _IndexFieldNumbers:
    metadata: int
    documents: int
    external_symbols: int


def load_proto_module(proto_module_path: Path) -> ScipProtoModule:
    """Load scip_pb2 from the generated file path.

    Returns
    -------
    ScipProtoModule
        Imported protobuf module.
    """
    module = load_generated_module(proto_module_path)
    return cast("ScipProtoModule", module)


def load_index(index_path: Path, proto_module_path: Path) -> IndexProto:
    """Load a SCIP index using generated protobuf bindings.

    Returns
    -------
    IndexProto
        Parsed protobuf Index instance.
    """
    module = load_proto_module(proto_module_path)
    index = module.Index()
    _parse_from_string(index, index_path.read_bytes())
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
    field_numbers = _index_field_numbers(index)

    text_document_encoding: str | None = None
    project_root: Path | None = None
    documents: list[ScipDocument] = []
    symbol_infos: list[ScipSymbolInfo] = []
    relationships: list[ScipSymbolRelationship] = []
    diagnostics: list[ScipDiagnostic] = []
    external_symbols: list[ScipExternalSymbol] = []

    for field_no, payload in _iter_index_payloads(index_path, field_numbers):
        if field_no == field_numbers.metadata:
            metadata = module.Metadata()
            _parse_from_string(metadata, payload)
            text_document_encoding = _normalize_text_document_encoding(
                getattr(metadata, "text_document_encoding", 0)
            )
            project_root = _parse_project_root(getattr(metadata, "project_root", None))
            continue
        if field_no == field_numbers.documents:
            doc = module.Document()
            _parse_from_string(doc, payload)
            position_encoding = _normalize_position_encoding(getattr(doc, "position_encoding", 0))
            documents.append(
                _parse_document(
                    doc,
                    position_encoding=position_encoding,
                    text_document_encoding=text_document_encoding,
                )
            )
            for sym_info in doc.symbols:
                symbol_infos.append(_parse_symbol_info(sym_info))
                relationships.extend(_parse_relationships(sym_info))
            diagnostics.extend(
                _parse_document_diagnostics(
                    module,
                    doc,
                    doc.relative_path,
                    position_encoding=position_encoding,
                    text_document_encoding=text_document_encoding,
                )
            )
            continue
        if field_no == field_numbers.external_symbols:
            sym = module.SymbolInformation()
            _parse_from_string(sym, payload)
            external_symbols.append(_parse_external_symbol(sym))

    return ScipParsedIndex(
        documents=tuple(documents),
        symbol_infos=tuple(symbol_infos),
        relationships=tuple(relationships),
        diagnostics=tuple(diagnostics),
        external_symbols=tuple(external_symbols),
        project_root=project_root,
    )


def rebase_parsed_index(parsed: ScipParsedIndex, repo_root: Path) -> ScipParsedIndex:
    """Rebase document and diagnostic paths to repo_root."""
    project_root = parsed.project_root
    if project_root is None:
        return parsed
    project_root = project_root.resolve()
    repo_root = repo_root.resolve()
    if project_root == repo_root:
        return parsed

    rebased_docs: list[ScipDocument] = []
    rel_path_map: dict[str, str] = {}
    docs_changed = False
    for doc in parsed.documents:
        rebased = _rebase_rel_path(project_root, repo_root, doc.relative_path)
        if rebased is None or rebased == doc.relative_path:
            rebased_docs.append(doc)
            continue
        rel_path_map[doc.relative_path] = rebased
        docs_changed = True
        rebased_docs.append(
            ScipDocument(
                relative_path=rebased,
                symbols=doc.symbols,
                occurrences=doc.occurrences,
                position_encoding=doc.position_encoding,
                text_document_encoding=doc.text_document_encoding,
            )
        )

    rebased_diagnostics: list[ScipDiagnostic] = []
    diagnostics_changed = False
    for diag in parsed.diagnostics:
        rebased = rel_path_map.get(diag.rel_path)
        if rebased is None:
            rebased = _rebase_rel_path(project_root, repo_root, diag.rel_path)
        if rebased is None or rebased == diag.rel_path:
            rebased_diagnostics.append(diag)
            continue
        diagnostics_changed = True
        rebased_diagnostics.append(
            ScipDiagnostic(
                rel_path=rebased,
                start_line=diag.start_line,
                start_col=diag.start_col,
                end_line=diag.end_line,
                end_col=diag.end_col,
                position_encoding=diag.position_encoding,
                text_document_encoding=diag.text_document_encoding,
                severity=diag.severity,
                code=diag.code,
                message=diag.message,
                source=diag.source,
            )
        )

    if not docs_changed and not diagnostics_changed:
        return parsed

    return ScipParsedIndex(
        documents=tuple(rebased_docs),
        symbol_infos=parsed.symbol_infos,
        relationships=parsed.relationships,
        diagnostics=tuple(rebased_diagnostics),
        external_symbols=parsed.external_symbols,
        project_root=parsed.project_root,
    )


def _parse_project_root(value: str | None) -> Path | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
    else:
        path = unquote(value)
    if not path:
        return None
    return Path(path)


def _rebase_rel_path(project_root: Path, repo_root: Path, rel_path: str) -> str | None:
    abs_path = project_root / rel_path
    try:
        rebased = abs_path.relative_to(repo_root)
    except ValueError:
        return None
    return rebased.as_posix()


def _parse_document(
    doc: DocumentProto,
    *,
    position_encoding: int | None,
    text_document_encoding: str | None,
) -> ScipDocument:
    symbols = tuple(_parse_symbol(sym) for sym in doc.symbols)
    occurrences_list: list[ScipOccurrence] = []
    for occ in doc.occurrences:
        parsed = _parse_occurrence(
            occ,
            position_encoding=position_encoding,
            text_document_encoding=text_document_encoding,
        )
        if parsed is not None:
            occurrences_list.append(parsed)
    return ScipDocument(
        relative_path=doc.relative_path,
        symbols=symbols,
        occurrences=tuple(occurrences_list),
        position_encoding=position_encoding,
        text_document_encoding=text_document_encoding,
    )


def _parse_symbol(sym: SymbolInfoProto) -> ScipSymbol:
    documentation = "\n".join(sym.documentation) if sym.documentation else None
    return ScipSymbol(symbol=sym.symbol, documentation=documentation)


def _parse_occurrence(
    occ: OccurrenceProto,
    *,
    position_encoding: int | None,
    text_document_encoding: str | None,
) -> ScipOccurrence | None:
    range_tuple = _parse_range(occ.range)
    if range_tuple is None:
        return None
    return ScipOccurrence(
        symbol=occ.symbol,
        range_start_line=range_tuple[0],
        range_start_col=range_tuple[1],
        range_end_line=range_tuple[2],
        range_end_col=range_tuple[3],
        symbol_roles=occ.symbol_roles,
        position_encoding=position_encoding,
        text_document_encoding=text_document_encoding,
        start_byte=None,
        end_byte=None,
    )


def _parse_range(rng: IntListProto) -> tuple[int, int, int, int] | None:
    length = len(rng)
    if length == _RANGE_LEN_SAME_LINE:
        return (int(rng[0]), int(rng[1]), int(rng[0]), int(rng[2]))
    if length == _RANGE_LEN_FULL:
        return (int(rng[0]), int(rng[1]), int(rng[2]), int(rng[3]))
    return None


def _parse_symbol_info(sym: SymbolInfoProto) -> ScipSymbolInfo:
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


def _signature_text(sym: SymbolInfoProto) -> str | None:
    sig_doc = sym.signature_documentation
    if sig_doc is None:
        return None
    text = sig_doc.text if hasattr(sig_doc, "text") else ""
    return text or None


def _parse_relationships(sym: SymbolInfoProto) -> list[ScipSymbolRelationship]:
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
    module: ScipProtoModule,
    doc: DocumentProto,
    rel_path: str,
    *,
    position_encoding: int | None,
    text_document_encoding: str | None,
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
                    position_encoding=position_encoding,
                    text_document_encoding=text_document_encoding,
                    severity=severity,
                    code=diag.code or None,
                    message=diag.message,
                    source=diag.source or None,
                )
            )
    return diagnostics


def _severity_name(module: ScipProtoModule, value: int) -> str:
    severity = getattr(module, "Severity", None)
    name_fn = getattr(severity, "Name", None)
    if not callable(name_fn):
        return "Unspecified"
    try:
        return str(name_fn(value))
    except (TypeError, ValueError):
        return "Unspecified"


def _parse_external_symbol(sym: ExternalSymbolProto) -> ScipExternalSymbol:
    manager, name, version = _parse_package_triple(sym.symbol)
    return ScipExternalSymbol(
        symbol=sym.symbol,
        package_manager=manager,
        package_name=name,
        package_version=version,
    )


def _parse_from_string(message: object, payload: bytes) -> None:
    parse_fn = getattr(message, "ParseFromString", None)
    if not callable(parse_fn):
        message = "SCIP protobuf ParseFromString is unavailable"
        raise TypeError(message)
    parse_fn(payload)


def _read_varint(handle: BinaryIO) -> int:
    shift = 0
    out = 0
    while True:
        raw = handle.read(1)
        if not raw:
            raise EOFError
        byte = raw[0]
        out |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return out
        shift += 7
        if shift >= _MAX_VARINT_SHIFT:
            message = "SCIP protobuf varint is too long"
            raise ValueError(message)


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise EOFError
    return payload


def _skip_field(handle: BinaryIO, wire_type: int) -> None:
    if wire_type == _WIRE_TYPE_VARINT:
        _read_varint(handle)
        return
    if wire_type == _WIRE_TYPE_64BIT:
        _read_exact(handle, 8)
        return
    if wire_type == _WIRE_TYPE_LENGTH:
        length = _read_varint(handle)
        _read_exact(handle, length)
        return
    if wire_type == _WIRE_TYPE_32BIT:
        _read_exact(handle, 4)
        return
    message = f"Unsupported wire_type={wire_type}"
    raise ValueError(message)


def _index_field_numbers(index: IndexProto) -> _IndexFieldNumbers:
    descriptor = getattr(index, "DESCRIPTOR", None)
    fields_by_name = getattr(descriptor, "fields_by_name", None)
    if not isinstance(fields_by_name, Mapping):
        fields = getattr(descriptor, "fields", None)
        if fields is not None:
            try:
                fields_by_name = {field.name: field for field in fields if hasattr(field, "name")}
            except TypeError:
                fields_by_name = None
    if not isinstance(fields_by_name, Mapping):
        message = "SCIP protobuf Index descriptor fields are unavailable"
        raise TypeError(message)
    return _IndexFieldNumbers(
        metadata=_field_number(fields_by_name, "metadata"),
        documents=_field_number(fields_by_name, "documents"),
        external_symbols=_field_number(fields_by_name, "external_symbols"),
    )


def _field_number(fields_by_name: Mapping[str, object], name: str) -> int:
    field = fields_by_name.get(name)
    number = getattr(field, "number", None)
    if not isinstance(number, int):
        message = f"SCIP protobuf field number missing for {name!r}"
        raise TypeError(message)
    return number


def _iter_index_payloads(
    index_path: Path,
    field_numbers: _IndexFieldNumbers,
) -> Iterator[tuple[int, bytes]]:
    fields = {field_numbers.metadata, field_numbers.documents, field_numbers.external_symbols}
    with index_path.open("rb") as handle:
        while True:
            try:
                tag = _read_varint(handle)
            except EOFError:
                return
            field_no = tag >> 3
            wire_type = tag & 0x7
            if wire_type != _WIRE_TYPE_LENGTH:
                _skip_field(handle, wire_type)
                continue
            length = _read_varint(handle)
            payload = _read_exact(handle, length)
            if field_no in fields:
                yield field_no, payload


def _normalize_position_encoding(value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized <= 0:
        return None
    return normalized


def _normalize_text_document_encoding(value: int | None) -> str | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized == _TEXT_DOCUMENT_ENCODING_UTF8:
        return "utf-8"
    if normalized == _TEXT_DOCUMENT_ENCODING_UTF16:
        return "utf-16"
    return None


def _parse_package_triple(symbol: str) -> tuple[str | None, str | None, str | None]:
    if symbol.startswith("local "):
        return None, None, None
    tokens = _split_tokens_with_double_space_escape(symbol)
    if len(tokens) < _PACKAGE_TOKEN_COUNT:
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
    "rebase_parsed_index",
]
