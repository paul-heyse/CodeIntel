"""Fake implementations for testing.

This package provides fake implementations for:
1. Ingestion tool runner and service (legacy)
2. Build system protocols (new pattern)

Example (new protocol-based fakes)
----------------------------------
>>> from tests._helpers.fakes import ProtocolFakeProviders
>>> from codeintel.build.context import ContextResources
>>>
>>> # Create fake providers
>>> providers = ProtocolFakeProviders.defaults()
>>>
>>> # Use in context
>>> resources = ContextResources(providers=providers)

Example (legacy tool fakes)
---------------------------
>>> from tests._helpers.fakes import FakeToolRunner, FakeToolService
>>> runner = FakeToolRunner(cache_dir=Path("/tmp/cache"))
"""

# Legacy tool fakes (for ingestion tests)
from tests._helpers.fakes.tools import (
    FakeToolRunner,
    FakeToolService,
    FakeToolServiceConfig,
)

# New protocol-based fakes (for build system tests)
from tests._helpers.fakes.fake_providers import (
    FakeCoverageCollector,
    FakeGitHistoryProvider,
    FakeProviders as ProtocolFakeProviders,
    FakeScipIndexer as ProtocolFakeScipIndexer,
    FakeTestReporter,
    FakeToolRunner as ProtocolFakeToolRunner,
    FakeTypeChecker,
)

__all__ = [
    # Legacy fakes
    "FakeToolRunner",
    "FakeToolService",
    "FakeToolServiceConfig",
    # Protocol-based fakes
    "FakeCoverageCollector",
    "FakeGitHistoryProvider",
    "FakeTestReporter",
    "FakeTypeChecker",
    "ProtocolFakeProviders",
    "ProtocolFakeScipIndexer",
    "ProtocolFakeToolRunner",
]
