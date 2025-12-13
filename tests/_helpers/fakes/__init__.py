"""Fake implementations for testing.

This package provides fake implementations for:
1. Ingestion tool runner and service (legacy)
2. Build system protocols (new pattern)

Example (new protocol-based fakes)
----------------------------------
>>> from tests._helpers.fakes import ProtocolFakeProviders
>>> from codeintel.build.context import ContextResources
>>>
>>>
>>> providers = ProtocolFakeProviders.defaults()
>>>
>>>
>>> resources = ContextResources(providers=providers)

Example (legacy tool fakes)
---------------------------
>>> from tests._helpers.fakes import FakeToolRunner, FakeToolService
>>> runner = FakeToolRunner(cache_dir=Path("/tmp/cache"))
"""

from tests._helpers.fakes.fake_providers import (
    FakeCoverageCollector,
    FakeGitHistoryProvider,
    FakeTestReporter,
    FakeTypeChecker,
)
from tests._helpers.fakes.fake_providers import (
    FakeProviders as ProtocolFakeProviders,
)
from tests._helpers.fakes.fake_providers import (
    FakeScipIndexer as ProtocolFakeScipIndexer,
)
from tests._helpers.fakes.fake_providers import (
    FakeToolRunner as ProtocolFakeToolRunner,
)
from tests._helpers.fakes.serving_backends import (
    BackendHandle,
    ServingBackend,
    build_serving_backend,
)
from tests._helpers.fakes.tools import (
    FakeToolRunner,
    FakeToolService,
    FakeToolServiceConfig,
)
from tests._helpers.fakes.utilities import utcnow

__all__ = [
    "BackendHandle",
    "FakeCoverageCollector",
    "FakeGitHistoryProvider",
    "FakeTestReporter",
    "FakeToolRunner",
    "FakeToolService",
    "FakeToolServiceConfig",
    "FakeTypeChecker",
    "ProtocolFakeProviders",
    "ProtocolFakeScipIndexer",
    "ProtocolFakeToolRunner",
    "ServingBackend",
    "build_serving_backend",
    "utcnow",
]
