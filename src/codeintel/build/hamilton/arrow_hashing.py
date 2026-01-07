"""Arrow hashing helpers for Hamilton cache versioning."""

from __future__ import annotations

import hashlib
from collections.abc import Buffer
from io import RawIOBase

import pyarrow as pa
from hamilton.caching import fingerprinting
from pyarrow import ipc

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.exports.arrow_ipc import build_ipc_write_options


class _HashSink(RawIOBase):
    """Write-only stream that updates a hash digest."""

    def __init__(self, hasher: hashlib._Hash) -> None:
        self._hasher = hasher

    @staticmethod
    def writable() -> bool:
        return True

    def write(self, b: Buffer) -> int:
        view = memoryview(b)
        self._hasher.update(view)
        return view.nbytes


def _hash_table(table: pa.Table) -> str:
    hasher = hashlib.sha256()
    sink = _HashSink(hasher)
    options = build_ipc_write_options(
        compression=None,
        use_threads=True,
        unify_dictionaries=True,
        metadata_version="V5",
    )
    with pa.output_stream(sink) as output:
        writer = ipc.new_stream(output, table.schema, options=options)
        for batch in table.to_batches(max_chunksize=DEFAULT_ARROW_BATCH_SIZE):
            writer.write_batch(batch)
        writer.close()
    return hasher.hexdigest()


def register_arrow_hashing() -> None:
    """Register deterministic hashing for Arrow tables."""

    @fingerprinting.hash_value.register(pa.Table)
    def _hash_pa_table(
        value: pa.Table,
        *args: object,
        **kwargs: object,
    ) -> str:
        _ = (args, kwargs)
        return _hash_table(value)


register_arrow_hashing()
