"""Helpers for generating SCIP protobuf artifacts in tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.ingestion.scip.proto import load_generated_module

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import ModuleType

_PROTO_CACHE: dict[Path, Path] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def proto_source_path() -> Path:
    """Return the path to the scip.proto source file."""
    return (
        _repo_root() / "src" / "codeintel" / "ingestion" / "scip" / "proto" / "scip.proto"
    )


def ensure_proto_module(tmp_path: Path | None = None) -> Path:
    """Ensure a scip_pb2.py module is generated and return its path."""
    out_dir = tmp_path / "scip_proto" if tmp_path is not None else Path(
        tempfile.mkdtemp(prefix="scip_proto_")
    )
    cached = _PROTO_CACHE.get(out_dir)
    if cached is not None and cached.is_file():
        return cached

    proto_path = proto_source_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I",
            str(proto_path.parent),
            "--python_out",
            str(out_dir),
            str(proto_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "grpc_tools.protoc failed"
        raise RuntimeError(message)

    module_path = out_dir / "scip_pb2.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"scip_pb2.py was not generated at {module_path}")
    _PROTO_CACHE[out_dir] = module_path
    return module_path


def load_proto_module(proto_module_path: Path) -> ModuleType:
    """Load the generated scip_pb2 module."""
    return load_generated_module(proto_module_path)


def write_scip_index(
    output_path: Path,
    *,
    proto_module_path: Path,
    documents: Iterable[Mapping[str, Any]] | None = None,
    external_symbols: Iterable[str] | None = None,
) -> Path:
    """Write a SCIP index with minimal document payloads."""
    module = load_generated_module(proto_module_path)
    index = module.Index()

    for doc in documents or _default_documents():
        rel_path = _doc_path(doc)
        if not rel_path:
            continue
        doc_msg = index.documents.add()
        doc_msg.relative_path = rel_path
        for symbol_payload in doc.get("symbols", []):
            symbol = symbol_payload.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                continue
            sym_msg = doc_msg.symbols.add()
            sym_msg.symbol = symbol
            documentation = symbol_payload.get("documentation")
            if isinstance(documentation, str):
                sym_msg.documentation.append(documentation)
            elif isinstance(documentation, list):
                sym_msg.documentation.extend(str(item) for item in documentation)
        for occurrence_payload in doc.get("occurrences", []):
            symbol = occurrence_payload.get("symbol")
            rng = occurrence_payload.get("range")
            if not isinstance(symbol, str) or not isinstance(rng, list):
                continue
            occ_msg = doc_msg.occurrences.add()
            occ_msg.symbol = symbol
            occ_msg.range.extend(int(value) for value in rng)
            symbol_roles = occurrence_payload.get("symbol_roles")
            if isinstance(symbol_roles, int):
                occ_msg.symbol_roles = symbol_roles

    for symbol in external_symbols or ():
        if not isinstance(symbol, str) or not symbol:
            continue
        ext = index.external_symbols.add()
        ext.symbol = symbol

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(index.SerializeToString())
    return output_path


def _doc_path(doc: Mapping[str, Any]) -> str | None:
    rel_path = doc.get("relative_path")
    if isinstance(rel_path, str) and rel_path:
        return rel_path
    rel_path = doc.get("relativePath")
    if isinstance(rel_path, str) and rel_path:
        return rel_path
    return None


def _default_documents() -> list[Mapping[str, Any]]:
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


__all__ = [
    "ensure_proto_module",
    "load_proto_module",
    "proto_source_path",
    "write_scip_index",
]
