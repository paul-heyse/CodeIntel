"""Timezone-aware datetime utilities for storage operations.

This module provides shared time utilities used across tracking modules.
All timestamps are UTC-aware following the project's datetime hygiene rules.

Example
-------
>>> from codeintel.storage.helpers.time import utc_now
>>> ts = utc_now()
>>> ts.tzinfo is not None
True
"""

from __future__ import annotations

from codeintel.core.time import utc_now

__all__ = ["utc_now"]
