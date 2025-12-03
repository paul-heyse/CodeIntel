"""Tests for infrastructure logging utilities.

This module tests the logging utilities for ingestion operations including
the ChangeLogFileHandler, log_progress function, and get_change_logger.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from codeintel.ingestion.infrastructure_utilities.logging import (
    ChangeLogFileHandler,
    get_change_logger,
    log_progress,
)

# Test constants for log_progress
TEST_OP = "ingest"
TEST_SCOPE = "test/repo@abc123"
TEST_TABLE = "core.modules"
TEST_ROW_COUNT = 100
TEST_DURATION = 1.5


# =============================================================================
# ChangeLogFileHandler Tests
# =============================================================================


def test_change_log_handler_initializes_with_file(tmp_path: Path) -> None:
    """ChangeLogFileHandler should initialize with filename and set flag."""
    log_file = tmp_path / "change.log"
    handler = ChangeLogFileHandler(str(log_file))

    try:
        assert handler.codeintel_change_log is True
        assert handler.encoding == "utf-8"
    finally:
        handler.close()


def test_change_log_handler_can_log_messages(tmp_path: Path) -> None:
    """ChangeLogFileHandler should write log messages to file."""
    log_file = tmp_path / "change.log"
    handler = ChangeLogFileHandler(str(log_file))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    # Create a test logger with our handler
    test_logger = logging.getLogger("test.change_handler")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    try:
        test_logger.info("Test change detection message")
        handler.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "Test change detection message" in content
    finally:
        test_logger.removeHandler(handler)
        handler.close()


def test_change_log_handler_creates_file_on_log(tmp_path: Path) -> None:
    """ChangeLogFileHandler should create the log file when logging."""
    log_file = tmp_path / "new_change.log"
    assert not log_file.exists()

    handler = ChangeLogFileHandler(str(log_file))
    handler.setLevel(logging.INFO)

    test_logger = logging.getLogger("test.file_creation")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    try:
        test_logger.info("Creating file")
        handler.flush()

        assert log_file.exists()
    finally:
        test_logger.removeHandler(handler)
        handler.close()


# =============================================================================
# log_progress Tests
# =============================================================================


def test_log_progress_logs_structured_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_progress should emit a structured log message."""
    with caplog.at_level(
        logging.INFO, logger="codeintel.ingestion.infrastructure_utilities.logging"
    ):
        log_progress(
            op=TEST_OP,
            scope=TEST_SCOPE,
            table=TEST_TABLE,
            rows=TEST_ROW_COUNT,
            duration_s=TEST_DURATION,
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    message = record.message

    assert TEST_OP in message
    assert f"scope={TEST_SCOPE}" in message
    assert f"table={TEST_TABLE}" in message
    assert f"rows={TEST_ROW_COUNT}" in message
    assert "duration=1.50s" in message


def test_log_progress_with_zero_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_progress should handle zero rows."""
    with caplog.at_level(
        logging.INFO, logger="codeintel.ingestion.infrastructure_utilities.logging"
    ):
        log_progress(
            op="delete",
            scope="repo@commit",
            table="core.empty",
            rows=0,
            duration_s=0.01,
        )

    assert len(caplog.records) == 1
    assert "rows=0" in caplog.records[0].message


def test_log_progress_with_large_row_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_progress should handle large row counts."""
    large_count = 1_000_000
    with caplog.at_level(
        logging.INFO, logger="codeintel.ingestion.infrastructure_utilities.logging"
    ):
        log_progress(
            op="bulk_insert",
            scope="repo@commit",
            table="analytics.data",
            rows=large_count,
            duration_s=120.5,
        )

    assert len(caplog.records) == 1
    assert f"rows={large_count}" in caplog.records[0].message


# =============================================================================
# get_change_logger Tests
# =============================================================================


def test_get_change_logger_without_env_var() -> None:
    """get_change_logger should return logger without file handler when env not set."""
    # Ensure env var is not set
    original = os.environ.pop("CODEINTEL_CHANGE_LOG", None)

    try:
        logger = get_change_logger()

        assert logger.name == "codeintel.ingestion.change"
        assert logger.level == logging.INFO
        # Should not have ChangeLogFileHandler
        change_handlers = [h for h in logger.handlers if isinstance(h, ChangeLogFileHandler)]
        assert len(change_handlers) == 0
    finally:
        if original is not None:
            os.environ["CODEINTEL_CHANGE_LOG"] = original


def test_get_change_logger_with_env_var(tmp_path: Path) -> None:
    """get_change_logger should add file handler when env var is set."""
    log_file = tmp_path / "change_env.log"
    original = os.environ.get("CODEINTEL_CHANGE_LOG")
    os.environ["CODEINTEL_CHANGE_LOG"] = str(log_file)

    try:
        # Clear any existing handlers from previous tests
        logger = logging.getLogger("codeintel.ingestion.change")
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)

        logger = get_change_logger()

        assert logger.name == "codeintel.ingestion.change"
        change_handlers = [h for h in logger.handlers if isinstance(h, ChangeLogFileHandler)]
        assert len(change_handlers) == 1

        handler = change_handlers[0]
        assert handler.codeintel_change_log is True
        assert handler.level == logging.INFO
    finally:
        # Cleanup
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)
        if original is not None:
            os.environ["CODEINTEL_CHANGE_LOG"] = original
        else:
            os.environ.pop("CODEINTEL_CHANGE_LOG", None)


def test_get_change_logger_does_not_add_duplicate_handlers(tmp_path: Path) -> None:
    """get_change_logger should not add duplicate file handlers."""
    log_file = tmp_path / "change_dup.log"
    original = os.environ.get("CODEINTEL_CHANGE_LOG")
    os.environ["CODEINTEL_CHANGE_LOG"] = str(log_file)

    try:
        # Clear any existing handlers
        logger = logging.getLogger("codeintel.ingestion.change")
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)

        # Call multiple times
        logger1 = get_change_logger()
        logger2 = get_change_logger()
        logger3 = get_change_logger()

        assert logger1 is logger2 is logger3

        change_handlers = [h for h in logger3.handlers if isinstance(h, ChangeLogFileHandler)]
        # Should only be one handler despite multiple calls
        assert len(change_handlers) == 1
    finally:
        # Cleanup
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)
        if original is not None:
            os.environ["CODEINTEL_CHANGE_LOG"] = original
        else:
            os.environ.pop("CODEINTEL_CHANGE_LOG", None)


def test_get_change_logger_writes_to_file(tmp_path: Path) -> None:
    """get_change_logger should actually write to the log file."""
    log_file = tmp_path / "change_write.log"
    original = os.environ.get("CODEINTEL_CHANGE_LOG")
    os.environ["CODEINTEL_CHANGE_LOG"] = str(log_file)

    try:
        # Clear any existing handlers
        logger = logging.getLogger("codeintel.ingestion.change")
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)

        logger = get_change_logger()
        logger.info("Test message for file write")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "Test message for file write" in content
    finally:
        # Cleanup
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)
        if original is not None:
            os.environ["CODEINTEL_CHANGE_LOG"] = original
        else:
            os.environ.pop("CODEINTEL_CHANGE_LOG", None)


def test_get_change_logger_propagates_messages(tmp_path: Path) -> None:
    """get_change_logger should have propagate=True when file handler added."""
    log_file = tmp_path / "test_propagate.log"
    original = os.environ.get("CODEINTEL_CHANGE_LOG")
    os.environ["CODEINTEL_CHANGE_LOG"] = str(log_file)

    try:
        # Clear any existing handlers
        logger = logging.getLogger("codeintel.ingestion.change")
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)

        logger = get_change_logger()

        # When file handler is added, propagate should be True
        assert logger.propagate is True
    finally:
        # Cleanup
        for handler in list(logger.handlers):
            if isinstance(handler, ChangeLogFileHandler):
                handler.close()
                logger.removeHandler(handler)
        if original is not None:
            os.environ["CODEINTEL_CHANGE_LOG"] = original
        else:
            os.environ.pop("CODEINTEL_CHANGE_LOG", None)
