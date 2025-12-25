"""Tool runner registry tests."""

from __future__ import annotations

from codeintel.observability.runtime_registry import (
    count_subprocesses,
    register_subprocess,
    snapshot_subprocesses,
    unregister_subprocess,
)


def test_register_unregister_subprocess() -> None:
    """Register/unregister should update registry contents."""
    pid = 91001
    register_subprocess(pid=pid, command="scip-python")
    try:
        records = snapshot_subprocesses()
        assert any(record.pid == pid for record in records)
        assert count_subprocesses() >= 1
    finally:
        unregister_subprocess(pid=pid)
    assert all(record.pid != pid for record in snapshot_subprocesses())


def test_snapshot_subprocesses_respects_limit() -> None:
    """Snapshot limit should cap returned records."""
    pid_a = 91002
    pid_b = 91003
    register_subprocess(pid=pid_a, command="tool-a")
    register_subprocess(pid=pid_b, command="tool-b")
    try:
        records = snapshot_subprocesses(limit=1)
        assert len(records) == 1
    finally:
        unregister_subprocess(pid=pid_a)
        unregister_subprocess(pid=pid_b)
