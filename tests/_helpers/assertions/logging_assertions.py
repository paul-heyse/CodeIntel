"""Logging assertions for observing production signals in tests."""

from __future__ import annotations

from collections.abc import Iterable
from logging import LogRecord


def assert_logged(
    records: Iterable[LogRecord],
    *,
    level: str | None = None,
    containing: str | None = None,
) -> None:
    r"""
    Assert that a log record matching the criteria exists.

    Parameters
    ----------
    records
        Iterable of log records (e.g., caplog.records).
    level
        Optional level name to filter (e.g., \"WARNING\").
    containing
        Optional substring that must appear in the message.

    Raises
    ------
    AssertionError
        If no matching record is found.
    """
    for record in records:
        if level is not None and record.levelname != level:
            continue
        if containing is not None and containing not in record.getMessage():
            continue
        return

    criteria = f"level={level!r} containing={containing!r}"
    error_message = f"Expected log record matching {criteria} but found none"
    raise AssertionError(error_message)


__all__ = ["assert_logged"]
