"""Utility functions for fake implementations.

This module provides utility functions used across fake implementations.
"""

from __future__ import annotations

from datetime import datetime


def utcnow() -> datetime:
    """
    Return timezone-aware now for deterministic tests.

    Returns
    -------
    datetime
        Current timezone-aware datetime.
    """
    return datetime.now().astimezone()


__all__ = ["utcnow"]
