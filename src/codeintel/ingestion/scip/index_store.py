"""Index merge utilities for incremental SCIP updates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.ingestion.scip.proto import load_generated_module
from codeintel.ingestion.scip.proto_types import (
    DocumentProto,
    ExternalSymbolListProto,
    ExternalSymbolProto,
    IndexProto,
    ScipProtoModule,
    SymbolInfoProto,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def load_index_proto(index_path: Path, *, proto_module_path: Path) -> IndexProto:
    """Load a SCIP index using generated protobuf bindings.

    Returns
    -------
    IndexProto
        Parsed protobuf Index instance.
    """
    module = cast("ScipProtoModule", load_generated_module(proto_module_path))
    index = module.Index()
    _parse_from_string(index, index_path.read_bytes())
    return index


def write_index_proto(index: IndexProto, output_path: Path) -> None:
    """Write a SCIP index to disk atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_bytes(_serialize_to_string(index))
    tmp_path.replace(output_path)


def merge_indexes(
    *,
    base_index: IndexProto,
    shard_indexes: Sequence[IndexProto],
    deleted_paths: Iterable[str],
    proto_module_path: Path,
) -> IndexProto:
    """Merge changed shard indexes into a base index.

    Returns
    -------
    IndexProto
        Merged protobuf Index instance.
    """
    module = cast("ScipProtoModule", load_generated_module(proto_module_path))
    merged = module.Index()

    if _has_metadata(base_index):
        _copy_from(merged.metadata, base_index.metadata)
    elif shard_indexes:
        _copy_from(merged.metadata, shard_indexes[0].metadata)

    docs_by_path: dict[str, DocumentProto] = {
        doc.relative_path: doc for doc in base_index.documents
    }
    for rel_path in deleted_paths:
        docs_by_path.pop(rel_path, None)
    for shard in shard_indexes:
        for doc in shard.documents:
            existing = docs_by_path.get(doc.relative_path)
            if existing is not None:
                docs_by_path[doc.relative_path] = _merge_document_symbols(existing, doc)
            else:
                docs_by_path[doc.relative_path] = doc

    for rel_path in sorted(docs_by_path):
        doc_msg = merged.documents.add()
        _copy_from(doc_msg, docs_by_path[rel_path])

    merged_external = _merge_external_symbols(
        base_index.external_symbols,
        (shard.external_symbols for shard in shard_indexes),
    )
    for symbol in sorted(merged_external):
        ext_msg = merged.external_symbols.add()
        _copy_from(ext_msg, merged_external[symbol])

    return merged


def _has_metadata(index: IndexProto) -> bool:
    has_field = getattr(index, "HasField", None)
    if not callable(has_field):
        return False
    return bool(has_field("metadata"))


def _merge_external_symbols(
    base_symbols: ExternalSymbolListProto,
    shard_symbol_iters: Iterable[ExternalSymbolListProto],
) -> dict[str, ExternalSymbolProto]:
    merged: dict[str, ExternalSymbolProto] = {sym.symbol: sym for sym in base_symbols}
    for shard_symbols in shard_symbol_iters:
        for sym in shard_symbols:
            existing = merged.get(sym.symbol)
            if existing is None or _prefer_external_symbol(existing, sym) is sym:
                merged[sym.symbol] = sym
    return merged


def _merge_document_symbols(
    base_doc: DocumentProto,
    shard_doc: DocumentProto,
) -> DocumentProto:
    base_symbols = {sym.symbol: sym for sym in base_doc.symbols}
    shard_symbols = {sym.symbol: sym for sym in shard_doc.symbols}
    merged: dict[str, SymbolInfoProto] = {}
    for symbol, shard_sym in shard_symbols.items():
        base_sym = base_symbols.get(symbol)
        merged[symbol] = _prefer_symbol_info(base_sym, shard_sym)

    _clear_field(shard_doc, "symbols")
    for symbol in sorted(merged):
        sym_msg = shard_doc.symbols.add()
        _copy_from(sym_msg, merged[symbol])
    return shard_doc


def _prefer_symbol_info(
    base_sym: SymbolInfoProto | None,
    shard_sym: SymbolInfoProto,
) -> SymbolInfoProto:
    if base_sym is None:
        return shard_sym
    base_score = _symbol_info_score(base_sym)
    shard_score = _symbol_info_score(shard_sym)
    if shard_score > base_score:
        return shard_sym
    if shard_score < base_score:
        return base_sym
    return shard_sym


def _symbol_info_score(sym: SymbolInfoProto) -> int:
    score = 0
    if _has_non_empty_strings(sym.documentation):
        score += 1
    if len(sym.relationships) > 0:
        score += 1
    if int(sym.kind) > 0:
        score += 1
    if sym.display_name:
        score += 1
    if _signature_text(sym):
        score += 1
    if sym.enclosing_symbol:
        score += 1
    return score


def _has_non_empty_strings(values: Iterable[str]) -> bool:
    return any(text.strip() for text in values if isinstance(text, str))


def _signature_text(sym: SymbolInfoProto) -> str:
    sig_doc = sym.signature_documentation
    if sig_doc is None:
        return ""
    text = getattr(sig_doc, "text", "")
    if not isinstance(text, str):
        return ""
    return text


def _prefer_external_symbol(
    base_sym: ExternalSymbolProto,
    shard_sym: ExternalSymbolProto,
) -> ExternalSymbolProto:
    base_score = _external_symbol_score(base_sym.symbol)
    shard_score = _external_symbol_score(shard_sym.symbol)
    if shard_score > base_score:
        return shard_sym
    if shard_score < base_score:
        return base_sym
    return shard_sym


def _external_symbol_score(symbol: str) -> int:
    manager, name, version = _parse_package_triple(symbol)
    return sum(1 for value in (manager, name, version) if value)


def _parse_from_string(index: IndexProto, payload: bytes) -> None:
    parse_fn = getattr(index, "ParseFromString", None)
    if not callable(parse_fn):
        message = "SCIP protobuf ParseFromString is unavailable"
        raise TypeError(message)
    parse_fn(payload)


def _serialize_to_string(index: IndexProto) -> bytes:
    serialize_fn = getattr(index, "SerializeToString", None)
    if not callable(serialize_fn):
        message = "SCIP protobuf SerializeToString is unavailable"
        raise TypeError(message)
    data = serialize_fn()
    if not isinstance(data, (bytes, bytearray)):
        message = "SCIP protobuf SerializeToString returned invalid data"
        raise TypeError(message)
    return bytes(data)


def _copy_from(target: object, source: object) -> None:
    copy_fn = getattr(target, "CopyFrom", None)
    if not callable(copy_fn):
        message = "SCIP protobuf CopyFrom is unavailable"
        raise TypeError(message)
    copy_fn(source)


def _clear_field(target: object, field_name: str) -> None:
    clear_fn = getattr(target, "ClearField", None)
    if not callable(clear_fn):
        return
    clear_fn(field_name)


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


_PACKAGE_TOKEN_COUNT = 4


__all__ = [
    "load_index_proto",
    "merge_indexes",
    "write_index_proto",
]
