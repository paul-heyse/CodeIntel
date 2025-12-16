"""Tests for PR-96: safe Hamilton parallel execution.

These tests validate that the threadpool execution adapter:
1) returns dict outputs (not pandas DataFrames)
2) serializes materialize/artifact nodes under a global write lock
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import TYPE_CHECKING

from hamilton.driver import Driver
from hamilton.function_modifiers import tag

from codeintel.build.hamilton import tags as ht
from codeintel.build.hamilton.adapters.parallel import ThreadPoolAdapter
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_is_instance

if TYPE_CHECKING:
    from collections.abc import Callable

    from hamilton.driver import HamiltonNode

    _FinalVar = str | HamiltonNode | Callable[..., object]


def test_pr96_threadpool_write_lock_serializes_materialize_nodes() -> None:
    """Ensure materialize-tagged nodes never execute concurrently."""
    counter_lock = threading.Lock()
    active_writes = 0
    max_active_writes = 0

    def seed() -> int:
        return 1

    def make_write_node(name: str) -> Callable[[int], int]:
        def node(seed: int) -> int:
            nonlocal active_writes
            nonlocal max_active_writes

            with counter_lock:
                active_writes += 1
                max_active_writes = max(max_active_writes, active_writes)

            time.sleep(0.05)

            with counter_lock:
                active_writes -= 1

            return seed

        tagged = tag(node_type=ht.NODE_TYPE_MATERIALIZE)(node)
        tagged.__name__ = name
        return tagged

    mod = types.ModuleType("codeintel_pr96_parallel_test")
    sys.modules[mod.__name__] = mod

    seed.__module__ = mod.__name__
    setattr(mod, seed.__name__, seed)

    outputs: list[_FinalVar] = []
    for i in range(4):
        fn = make_write_node(f"write_{i}")
        fn.__module__ = mod.__name__
        setattr(mod, fn.__name__, fn)
        outputs.append(fn.__name__)

    adapter = ThreadPoolAdapter(max_workers=4)
    dr = Driver({}, mod, adapter=[adapter])
    result = dr.execute(outputs)

    expect_is_instance(result, dict)
    expect_equal(max_active_writes, 1)
