"""Deprecated wrapper for analytics table helpers.

Use codeintel.build.tabular.frames instead.
"""

from __future__ import annotations

from codeintel.build.tabular.frames import ColumnsSpec, empty_frame_for_table, rows_to_frame

__all__ = ["ColumnsSpec", "empty_frame_for_table", "rows_to_frame"]
