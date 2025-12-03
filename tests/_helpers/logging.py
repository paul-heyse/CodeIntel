"""Logging helpers for tests."""

from __future__ import annotations

import logging
from typing import Final


class CapturingHandler(logging.Handler):
    """Logging handler that captures records for assertions."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append the record for later inspection."""
        self.records.append(record)


CAPTURE_HANDLER_LEVEL: Final[int] = logging.INFO
