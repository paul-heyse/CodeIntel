"""Tests for SCIP index merge policies."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.scip.index_store import (
    MergeIndexContext,
    load_index_proto,
    merge_indexes,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.scip_proto import ensure_proto_module, write_scip_index

if TYPE_CHECKING:
    from codeintel.ingestion.scip.proto_types import DocumentProto

SYMBOL_KIND_CLASS = 7


def _load_doc_symbol_text(doc: DocumentProto, symbol: str) -> tuple[str, str, str]:
    for sym in doc.symbols:
        if sym.symbol != symbol:
            continue
        documentation = "\n".join(sym.documentation) if sym.documentation else ""
        display_name = sym.display_name or ""
        signature_doc = sym.signature_documentation
        signature = signature_doc.text if signature_doc is not None else ""
        return documentation, display_name, signature
    message = f"Symbol {symbol} not found in merged document"
    raise AssertionError(message)


def test_merge_indexes_prefers_more_informative_symbol_info(tmp_path: Path) -> None:
    """merge_indexes should retain richer symbol_information when the shard regresses."""
    proto_module_path = ensure_proto_module(tmp_path)
    base_path = tmp_path / "base.scip"
    shard_path = tmp_path / "shard.scip"
    symbol = "scip-python python CodeIntel 0.1.0 `mod`/Foo#"

    write_scip_index(
        base_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "symbols": [
                    {
                        "symbol": symbol,
                        "documentation": "Base doc",
                        "display_name": "Foo",
                        "signature": "Foo()",
                        "kind": SYMBOL_KIND_CLASS,
                    }
                ],
            }
        ],
    )
    write_scip_index(
        shard_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "symbols": [
                    {
                        "symbol": symbol,
                    }
                ],
            }
        ],
    )

    base_index = load_index_proto(base_path, proto_module_path=proto_module_path)
    shard_index = load_index_proto(shard_path, proto_module_path=proto_module_path)
    merged = merge_indexes(
        base_index=base_index,
        shard_indexes=(shard_index,),
        deleted_paths=(),
        proto_module_path=proto_module_path,
    )

    doc = merged.documents[0]
    documentation, display_name, signature = _load_doc_symbol_text(doc, symbol)
    expect_true(documentation == "Base doc")
    expect_true(display_name == "Foo")
    expect_true(signature == "Foo()")


def test_merge_indexes_prefers_newer_symbol_info_when_richer(tmp_path: Path) -> None:
    """merge_indexes should adopt shard symbol info when it adds detail."""
    proto_module_path = ensure_proto_module(tmp_path)
    base_path = tmp_path / "base.scip"
    shard_path = tmp_path / "shard.scip"
    symbol = "scip-python python CodeIntel 0.1.0 `mod`/Bar#"
    base_updated_at = datetime(2024, 1, 1, tzinfo=UTC)
    shard_updated_at = datetime(2024, 2, 1, tzinfo=UTC)

    write_scip_index(
        base_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "symbols": [{"symbol": symbol}],
            }
        ],
    )
    write_scip_index(
        shard_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "symbols": [
                    {
                        "symbol": symbol,
                        "documentation": "Shard doc",
                        "display_name": "Bar",
                        "signature": "Bar()",
                        "kind": SYMBOL_KIND_CLASS,
                    }
                ],
            }
        ],
    )

    base_index = load_index_proto(base_path, proto_module_path=proto_module_path)
    shard_index = load_index_proto(shard_path, proto_module_path=proto_module_path)
    merged = merge_indexes(
        base_index=base_index,
        shard_indexes=(shard_index,),
        deleted_paths=(),
        proto_module_path=proto_module_path,
        context=MergeIndexContext(
            base_updated_at={"mod.py": base_updated_at},
            shard_updated_at={"mod.py": shard_updated_at},
        ),
    )

    doc = merged.documents[0]
    documentation, display_name, signature = _load_doc_symbol_text(doc, symbol)
    expect_true(documentation == "Shard doc")
    expect_true(display_name == "Bar")
    expect_true(signature == "Bar()")


def test_merge_indexes_retains_newer_base_when_shard_is_older(tmp_path: Path) -> None:
    """merge_indexes should keep base symbol info when shard is older."""
    proto_module_path = ensure_proto_module(tmp_path)
    base_path = tmp_path / "base.scip"
    shard_path = tmp_path / "shard.scip"
    symbol = "scip-python python CodeIntel 0.1.0 `mod`/Baz#"
    base_updated_at = datetime(2024, 3, 1, tzinfo=UTC)
    shard_updated_at = datetime(2024, 1, 1, tzinfo=UTC)

    write_scip_index(
        base_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "symbols": [{"symbol": symbol}],
            }
        ],
    )
    write_scip_index(
        shard_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "symbols": [
                    {
                        "symbol": symbol,
                        "documentation": "Shard doc",
                        "display_name": "Baz",
                        "signature": "Baz()",
                        "kind": SYMBOL_KIND_CLASS,
                    }
                ],
            }
        ],
    )

    base_index = load_index_proto(base_path, proto_module_path=proto_module_path)
    shard_index = load_index_proto(shard_path, proto_module_path=proto_module_path)
    merged = merge_indexes(
        base_index=base_index,
        shard_indexes=(shard_index,),
        deleted_paths=(),
        proto_module_path=proto_module_path,
        context=MergeIndexContext(
            base_updated_at={"mod.py": base_updated_at},
            shard_updated_at={"mod.py": shard_updated_at},
        ),
    )

    doc = merged.documents[0]
    documentation, display_name, signature = _load_doc_symbol_text(doc, symbol)
    expect_true(not documentation)
    expect_true(not display_name)
    expect_true(not signature)


def test_merge_indexes_dedupes_external_symbols(tmp_path: Path) -> None:
    """merge_indexes should dedupe external_symbols by symbol string."""
    proto_module_path = ensure_proto_module(tmp_path)
    base_path = tmp_path / "base.scip"
    shard_path = tmp_path / "shard.scip"

    write_scip_index(
        base_path,
        proto_module_path=proto_module_path,
        documents=[{"relative_path": "mod.py"}],
        external_symbols=[
            "scip-python python requests 2.31.0 `requests`/",
            "scip-python python numpy 1.26.0 `numpy`/",
        ],
    )
    write_scip_index(
        shard_path,
        proto_module_path=proto_module_path,
        documents=[{"relative_path": "mod.py"}],
        external_symbols=[
            "scip-python python numpy 1.26.0 `numpy`/",
            "scip-python python pandas 2.1.0 `pandas`/",
        ],
    )

    base_index = load_index_proto(base_path, proto_module_path=proto_module_path)
    shard_index = load_index_proto(shard_path, proto_module_path=proto_module_path)
    merged = merge_indexes(
        base_index=base_index,
        shard_indexes=(shard_index,),
        deleted_paths=(),
        proto_module_path=proto_module_path,
    )

    symbols = {sym.symbol for sym in merged.external_symbols}
    expect_equal(
        symbols,
        {
            "scip-python python requests 2.31.0 `requests`/",
            "scip-python python numpy 1.26.0 `numpy`/",
            "scip-python python pandas 2.1.0 `pandas`/",
        },
    )
