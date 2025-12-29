"""Tests for LazyFrame streaming fallback behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from polars.exceptions import PolarsError

from codeintel.core.columnar.stream import LazyFrameStream

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame

pl = pytest.importorskip("polars")

ROW_COUNT = 3

pytestmark = pytest.mark.no_runtime_env


def test_lazyframe_stream_uses_fallback_on_streaming_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure streaming fallback triggers on Polars streaming errors."""
    frame = pl.DataFrame({"value": [1, 2, ROW_COUNT]})
    lazyframe = frame.lazy()
    calls = {"streaming": 0, "fallback": 0}
    message = "streaming not supported"

    def fake_collect_batches(
        lazyframe: LazyFrame,
        *,
        batch_size: int,
        options: object,
    ) -> list[DataFrame]:
        _ = batch_size
        if getattr(options, "streaming", False):
            calls["streaming"] += 1
            raise PolarsError(message)
        calls["fallback"] += 1
        return [lazyframe.collect()]

    monkeypatch.setattr(
        "codeintel.core.columnar.stream.collect_batches",
        fake_collect_batches,
    )
    stream = LazyFrameStream(
        lazyframe=lazyframe,
        streaming=True,
        streaming_fallback=True,
    )
    reader = stream.to_reader(batch_size=2)
    batches = list(reader)

    assert sum(batch.num_rows for batch in batches) == ROW_COUNT
    assert calls["streaming"] == 1
    assert calls["fallback"] == 1
