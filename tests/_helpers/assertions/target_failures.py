"""Assertion helpers for TargetRunRecord failure scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.hamilton.run_records import TargetRunRecord


def assert_partial_failure(
    records: Mapping[str, TargetRunRecord],
    *,
    failed: Iterable[str],
    succeeded: Iterable[str] = (),
    skipped: Iterable[str] = (),
) -> None:
    """Assert a mixed TargetRunRecord bundle matches expected outcomes.

    Parameters
    ----------
    records
        Mapping of target name to TargetRunRecord.
    failed
        Target names expected to have status "failed".
    succeeded
        Target names expected to have status "succeeded".
    skipped
        Target names expected to have status "skipped".

    """
    _assert_statuses(records, failed, "failed")
    _assert_statuses(records, succeeded, "succeeded")
    _assert_statuses(records, skipped, "skipped")


def _assert_statuses(
    records: Mapping[str, TargetRunRecord],
    targets: Iterable[str],
    status: str,
) -> None:
    for target in targets:
        record = records.get(target)
        if record is None:
            message = f"Missing TargetRunRecord for {target}"
            raise AssertionError(message)
        expect_equal(record.status, status, label=target)


__all__ = [
    "assert_partial_failure",
]
