"""Cancellation helpers for serving operations."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

CancelCheck = Callable[[], None]


class OperationCancelledError(TimeoutError):
    """Raised when a serving operation is cancelled."""


class CancelToken:
    """Thread-safe cancellation token with optional deadline enforcement."""

    def __init__(self, *, deadline_s: float | None = None) -> None:
        self._event = threading.Event()
        self._deadline_s = deadline_s
        self._reason: str | None = None

    @classmethod
    def from_timeout(cls, timeout_s: float | None) -> CancelToken:
        """Create a cancellation token with an optional timeout deadline.

        Parameters
        ----------
        timeout_s
            Timeout in seconds. None disables the deadline.

        Returns
        -------
        CancelToken
            Cancellation token with an optional deadline.
        """
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        return cls(deadline_s=deadline)

    @property
    def cancelled(self) -> bool:
        """Return True when cancellation has been signaled.

        Returns
        -------
        bool
            True when cancellation has been requested.
        """
        return self._event.is_set()

    def cancel(self, *, reason: str | None = None) -> None:
        """Signal cancellation with an optional reason.

        Parameters
        ----------
        reason
            Optional reason to associate with the cancellation.
        """
        if reason:
            self._reason = reason
        self._event.set()

    def raise_if_cancelled(self) -> None:
        """Raise when cancellation or deadline is reached.

        Raises
        ------
        OperationCancelledError
            If cancellation has been requested.
        TimeoutError
            If the timeout deadline has been reached.
        """
        if self._event.is_set():
            msg = self._reason or "Operation cancelled"
            raise OperationCancelledError(msg)
        if self._deadline_s is not None and time.monotonic() >= self._deadline_s:
            msg = "Operation timed out"
            raise TimeoutError(msg)


__all__ = ["CancelCheck", "CancelToken", "OperationCancelledError"]
