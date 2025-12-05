"""Test harness infrastructure for plugin testing.

This package provides fluent test harnesses for both analytics and ingestion
plugins, with shared base classes to reduce code duplication.

Example
-------
>>> from tests._helpers.harnesses import PluginTestHarness, IngestPluginTestHarness
>>> from tests._helpers.harnesses import assert_result, assert_ingest_result
"""

from __future__ import annotations

from tests._helpers.harnesses.analytics import (
    PluginResultAssertions,
    PluginTestHarness,
    ValidationResultAssertions,
    assert_result,
    assert_validation,
)
from tests._helpers.harnesses.base import (
    BaseResultAssertions,
    BaseTestHarness,
    ResultLike,
)
from tests._helpers.harnesses.graphs import (
    GraphPluginTestHarness,
    NewPluginTestHarness,
)
from tests._helpers.harnesses.ingest_setup import IngestTestSetup
from tests._helpers.harnesses.ingestion import (
    IngestPluginResultAssertions,
    IngestPluginTestHarness,
    assert_ingest_result,
)

__all__ = [
    "BaseResultAssertions",
    "BaseTestHarness",
    "GraphPluginTestHarness",
    "IngestPluginResultAssertions",
    "IngestPluginTestHarness",
    "IngestTestSetup",
    "NewPluginTestHarness",
    "PluginResultAssertions",
    "PluginTestHarness",
    "ResultLike",
    "ValidationResultAssertions",
    "assert_ingest_result",
    "assert_result",
    "assert_validation",
]
