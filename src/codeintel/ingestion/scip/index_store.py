"""Index merge utilities for incremental SCIP updates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.scip.proto import load_generated_module

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def load_index_proto(index_path: Path, *, proto_module_path: Path) -> object:
    """Load a SCIP index using generated protobuf bindings.

    Returns
    -------
    object
        Parsed protobuf Index instance.
    """
    module = load_generated_module(proto_module_path)
    index = module.Index()
    index.ParseFromString(index_path.read_bytes())
    return index


def write_index_proto(index: object, output_path: Path) -> None:
    """Write a SCIP index to disk atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_bytes(index.SerializeToString())
    tmp_path.replace(output_path)


def merge_indexes(
    *,
    base_index: object,
    shard_indexes: Sequence[object],
    deleted_paths: Iterable[str],
    proto_module_path: Path,
) -> object:
    """Merge changed shard indexes into a base index.

    Returns
    -------
    object
        Merged protobuf Index instance.
    """
    module = load_generated_module(proto_module_path)
    merged = module.Index()

    if _has_metadata(base_index):
        merged.metadata.CopyFrom(base_index.metadata)
    elif shard_indexes:
        merged.metadata.CopyFrom(shard_indexes[0].metadata)

    docs_by_path = {doc.relative_path: doc for doc in base_index.documents}
    for rel_path in deleted_paths:
        docs_by_path.pop(rel_path, None)
    for shard in shard_indexes:
        for doc in shard.documents:
            docs_by_path[doc.relative_path] = doc

    for rel_path in sorted(docs_by_path):
        merged.documents.add().CopyFrom(docs_by_path[rel_path])

    merged_external = _merge_external_symbols(
        base_index.external_symbols,
        (shard.external_symbols for shard in shard_indexes),
    )
    for symbol in sorted(merged_external):
        merged.external_symbols.add().CopyFrom(merged_external[symbol])

    return merged


def _has_metadata(index: object) -> bool:
    has_field = getattr(index, "HasField", None)
    if callable(has_field):
        return bool(index.HasField("metadata"))
    return False


def _merge_external_symbols(
    base_symbols: Sequence[object],
    shard_symbol_iters: Iterable[Sequence[object]],
) -> dict[str, object]:
    merged: dict[str, object] = {sym.symbol: sym for sym in base_symbols}
    for shard_symbols in shard_symbol_iters:
        for sym in shard_symbols:
            merged[sym.symbol] = sym
    return merged


__all__ = [
    "load_index_proto",
    "merge_indexes",
    "write_index_proto",
]
