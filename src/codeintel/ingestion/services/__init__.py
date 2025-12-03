"""Ingestion services providing high-level operations.

This module exposes service classes that wrap port adapters for common
ingestion operations.
"""

from __future__ import annotations

from codeintel.ingestion.services.storage import IngestStorageService

__all__ = ["IngestStorageService"]
