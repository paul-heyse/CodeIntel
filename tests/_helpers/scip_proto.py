"""Helpers for generating SCIP protobuf artifacts in tests."""

from __future__ import annotations

import contextlib
import io
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from grpc_tools import protoc

from codeintel.ingestion.scip.proto import load_generated_module
from codeintel.ingestion.scip.proto_types import (
    DocumentProto,
    IndexProto,
    OccurrenceProto,
    ScipProtoModule,
    SymbolInfoProto,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

_PROTO_CACHE: dict[Path, Path] = {}
_SEVERITY_BY_NAME: dict[str, int] = {
    "unspecifiedseverity": 0,
    "unspecified": 0,
    "error": 1,
    "warning": 2,
    "information": 3,
    "info": 3,
    "hint": 4,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def proto_source_path() -> Path:
    """Return the path to the scip.proto source file.

    Returns
    -------
    Path
        Filesystem path to scip.proto.
    """
    return _repo_root() / "src" / "codeintel" / "ingestion" / "scip" / "proto" / "scip.proto"


def _run_protoc(proto_path: Path, out_dir: Path) -> None:
    stderr = io.StringIO()
    args = [
        "grpc_tools.protoc",
        "-I",
        str(proto_path.parent),
        "--python_out",
        str(out_dir),
        str(proto_path),
    ]
    with contextlib.redirect_stderr(stderr):
        result = protoc.main(args)
    if result != 0:
        message = stderr.getvalue().strip() or "grpc_tools.protoc failed"
        raise RuntimeError(message)


def ensure_proto_module(tmp_path: Path | None = None) -> Path:
    """Ensure a scip_pb2.py module is generated and return its path.

    Returns
    -------
    Path
        Filesystem path to the generated scip_pb2.py file.

    Raises
    ------
    FileNotFoundError
        If the generated module is missing on disk.
    """
    out_dir = (
        tmp_path / "scip_proto"
        if tmp_path is not None
        else Path(tempfile.mkdtemp(prefix="scip_proto_"))
    )
    cached = _PROTO_CACHE.get(out_dir)
    if cached is not None and cached.is_file():
        return cached

    proto_path = proto_source_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_protoc(proto_path, out_dir)

    module_path = out_dir / "scip_pb2.py"
    if not module_path.is_file():
        message = "scip_pb2.py was not generated at " + str(module_path)
        raise FileNotFoundError(message)
    _PROTO_CACHE[out_dir] = module_path
    return module_path


def load_proto_module(proto_module_path: Path) -> ModuleType:
    """Load the generated scip_pb2 module.

    Returns
    -------
    ModuleType
        Imported protobuf module.
    """
    return load_generated_module(proto_module_path)


def _build_index(proto_module_path: Path) -> IndexProto:
    module = cast("ScipProtoModule", load_generated_module(proto_module_path))
    return module.Index()


def write_scip_index(
    output_path: Path,
    *,
    proto_module_path: Path,
    documents: Iterable[Mapping[str, object]] | None = None,
    external_symbols: Iterable[str | Mapping[str, object]] | None = None,
) -> Path:
    """Write a SCIP index with minimal document payloads.

    Returns
    -------
    Path
        Path to the written index file.
    """
    index = _build_index(proto_module_path)
    _add_documents(index, documents or _default_documents())
    _add_external_symbols(index, external_symbols or ())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_serialize_index(index))
    return output_path


def _add_documents(index: IndexProto, documents: Iterable[Mapping[str, object]]) -> None:
    for doc in documents:
        rel_path = _doc_path(doc)
        if not rel_path:
            continue
        doc_msg = index.documents.add()
        doc_msg.relative_path = rel_path
        _add_symbols(doc_msg, doc.get("symbols"))
        _add_occurrences(doc_msg, doc.get("occurrences"))


def _add_symbols(doc_msg: DocumentProto, value: object) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        sym_msg = _build_symbol_message(doc_msg, item)
        if sym_msg is None:
            continue
        _add_symbol_documentation(sym_msg, item)
        _add_symbol_metadata(sym_msg, item)
        relationships = item.get("relationships")
        if isinstance(relationships, list):
            _add_relationships(sym_msg, relationships)


def _add_occurrences(doc_msg: DocumentProto, value: object) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        if not isinstance(item, Mapping):
            continue
        symbol = item.get("symbol")
        rng = item.get("range")
        if not isinstance(symbol, str) or not isinstance(rng, list):
            continue
        occ_msg = doc_msg.occurrences.add()
        occ_msg.symbol = symbol
        occ_msg.range.extend(int(range_value) for range_value in rng)
        symbol_roles = item.get("symbol_roles")
        if isinstance(symbol_roles, int):
            occ_msg.symbol_roles = symbol_roles
        diagnostics = item.get("diagnostics")
        if isinstance(diagnostics, list):
            _add_diagnostics(occ_msg, diagnostics)


def _build_symbol_message(
    doc_msg: DocumentProto,
    item: object,
) -> SymbolInfoProto | None:
    if not isinstance(item, Mapping):
        return None
    symbol = item.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        return None
    sym_msg = doc_msg.symbols.add()
    sym_msg.symbol = symbol
    return sym_msg


def _add_symbol_documentation(sym_msg: SymbolInfoProto, item: Mapping[str, object]) -> None:
    documentation = item.get("documentation")
    if isinstance(documentation, str):
        sym_msg.documentation.append(documentation)
    elif isinstance(documentation, list):
        sym_msg.documentation.extend(str(doc_item) for doc_item in documentation)


def _add_symbol_metadata(sym_msg: SymbolInfoProto, item: Mapping[str, object]) -> None:
    kind = item.get("kind")
    if isinstance(kind, int):
        sym_msg.kind = kind
    display_name = item.get("display_name")
    if isinstance(display_name, str):
        sym_msg.display_name = display_name
    signature = item.get("signature")
    if isinstance(signature, str):
        signature_doc = sym_msg.signature_documentation
        if signature_doc is not None:
            signature_doc.text = signature
    enclosing_symbol = item.get("enclosing_symbol")
    if isinstance(enclosing_symbol, str):
        sym_msg.enclosing_symbol = enclosing_symbol


def _add_relationships(sym_msg: SymbolInfoProto, relationships: list[object]) -> None:
    relationships_attr = getattr(sym_msg, "relationships", None)
    if relationships_attr is None:
        return
    for item in relationships:
        if not isinstance(item, Mapping):
            continue
        rel_symbol = item.get("symbol")
        if not isinstance(rel_symbol, str) or not rel_symbol:
            continue
        rel_msg = relationships_attr.add()
        rel_msg.symbol = rel_symbol
        if item.get("is_reference") is True:
            rel_msg.is_reference = True
        if item.get("is_implementation") is True:
            rel_msg.is_implementation = True
        if item.get("is_type_definition") is True:
            rel_msg.is_type_definition = True
        if item.get("is_definition") is True:
            rel_msg.is_definition = True


def _add_diagnostics(occ_msg: OccurrenceProto, diagnostics: list[object]) -> None:
    diagnostics_attr = getattr(occ_msg, "diagnostics", None)
    if diagnostics_attr is None:
        return
    for item in diagnostics:
        if not isinstance(item, Mapping):
            continue
        message = item.get("message")
        if not isinstance(message, str):
            continue
        diag_msg = diagnostics_attr.add()
        severity = _resolve_severity(item.get("severity"))
        if severity is not None:
            diag_msg.severity = severity
        code = item.get("code")
        if isinstance(code, str):
            diag_msg.code = code
        diag_msg.message = message
        source = item.get("source")
        if isinstance(source, str):
            diag_msg.source = source


def _add_external_symbols(index: IndexProto, symbols: Iterable[str | Mapping[str, object]]) -> None:
    for item in symbols:
        if isinstance(item, str):
            symbol = item
        elif isinstance(item, Mapping):
            symbol = item.get("symbol")
        else:
            continue
        if not isinstance(symbol, str) or not symbol:
            continue
        ext = index.external_symbols.add()
        ext.symbol = symbol


def _serialize_index(index: IndexProto) -> bytes:
    serialize_fn = getattr(index, "SerializeToString", None)
    if not callable(serialize_fn):
        message = "SCIP protobuf SerializeToString is unavailable"
        raise TypeError(message)
    data = serialize_fn()
    if not isinstance(data, (bytes, bytearray)):
        message = "SCIP protobuf SerializeToString returned invalid data"
        raise TypeError(message)
    return bytes(data)


def _doc_path(doc: Mapping[str, object]) -> str | None:
    rel_path = doc.get("relative_path")
    if isinstance(rel_path, str) and rel_path:
        return rel_path
    rel_path = doc.get("relativePath")
    if isinstance(rel_path, str) and rel_path:
        return rel_path
    return None


def _default_documents() -> list[Mapping[str, object]]:
    return [
        {
            "relative_path": "src/example.py",
            "symbols": [{"symbol": "scip-python python src/example foo()."}],
            "occurrences": [
                {
                    "symbol": "scip-python python src/example foo().",
                    "range": [1, 0, 1, 1],
                    "symbol_roles": 1,
                }
            ],
        }
    ]


def _resolve_severity(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if not key:
            return None
        return _SEVERITY_BY_NAME.get(key)
    return None


__all__ = [
    "ensure_proto_module",
    "load_proto_module",
    "proto_source_path",
    "write_scip_index",
]
