"""Fake implementations for unit tests.

This package provides fake implementations for:
1. Ingestion tool runner and service (legacy)
2. Build system protocols (new pattern)

Use real gateways/harnesses in integration tests to preserve production parity.

Example (new protocol-based fakes)
----------------------------------
>>> from tests._helpers.fakes import ProtocolFakeProviders
>>>
>>>
>>> providers = ProtocolFakeProviders.defaults()

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
from tests._helpers.fakes.tools import (
    FakeToolRunner,
    FakeToolService,
    FakeToolServiceConfig,
)
from tests._helpers.fakes.utilities import utcnow

__all__ = [
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
    "utcnow",
]
