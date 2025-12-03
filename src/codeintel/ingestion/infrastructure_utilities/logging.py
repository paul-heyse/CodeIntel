"""Logging utilities for ingestion operations.

This module provides logging handlers and utilities for structured
logging during ingestion operations.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


class ChangeLogFileHandler(logging.FileHandler):
    """File handler tagged for change-detection logging.

    Attributes
    ----------
    codeintel_change_log
        Flag indicating this is a change log handler.
    """

    codeintel_change_log: bool
    _codeintel_change_log: bool

    def __init__(self, filename: str) -> None:
        """Initialize the handler.

        Parameters
        ----------
        filename
            Path to log file.
        """
        super().__init__(filename, encoding="utf-8")
        self.codeintel_change_log = True
        self._codeintel_change_log = True


def log_progress(op: str, *, scope: str, table: str, rows: int, duration_s: float) -> None:
    """Emit a structured ingest log entry.

    Parameters
    ----------
    op
        Operation name (e.g., "ingest").
    scope
        Scope identifier (e.g., "repo@commit").
    table
        Table name.
    rows
        Number of rows processed.
    duration_s
        Duration in seconds.
    """
    log.info(
        "%s scope=%s table=%s rows=%d duration=%.2fs",
        op,
        scope,
        table,
        rows,
        duration_s,
    )


def get_change_logger() -> logging.Logger:
    """Return a logger that also writes to a file when configured.

    Set CODEINTEL_CHANGE_LOG to a file path to enable persistent logging
    of change detection decisions.

    Returns
    -------
    logging.Logger
        Logger configured for change detection diagnostics.
    """
    logger = logging.getLogger("codeintel.ingestion.change")
    logger.setLevel(logging.INFO)
    log_path = os.getenv("CODEINTEL_CHANGE_LOG")
    if log_path:
        existing = any(isinstance(handler, ChangeLogFileHandler) for handler in logger.handlers)
        if not existing:
            handler = ChangeLogFileHandler(log_path)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = True
    return logger


__all__ = [
    "ChangeLogFileHandler",
    "get_change_logger",
    "log_progress",
]
