"""Tests for schema registry producer/consumer resolution."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
from codeintel.build.hamilton.native.analytics.hotspots import HOTSPOTS_TARGET_NAME
from codeintel.build.hamilton.native.ingestion.ingest_targets import (
    MODULES_TABLE_KEY,
    MODULES_TARGET_NAME,
)


def test_schema_registry_producers_consumers_use_tags() -> None:
    """Producer/consumer lists should be derived from Hamilton tag metadata."""
    producers = SCHEMA_REGISTRY.producers_of(MODULES_TABLE_KEY)
    if MODULES_TARGET_NAME not in producers:
        pytest.fail("Expected modules target to be registered as a producer of core.modules.")

    consumers = SCHEMA_REGISTRY.consumers_of(MODULES_TABLE_KEY)
    if HOTSPOTS_TARGET_NAME not in consumers:
        pytest.fail("Expected hotspots target to be registered as a consumer of core.modules.")
