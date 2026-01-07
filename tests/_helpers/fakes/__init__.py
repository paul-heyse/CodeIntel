"""Fake implementations for testing.

This package provides fake implementations for build system protocols.

Example (new protocol-based fakes)
----------------------------------
>>> from tests._helpers.fakes import ProtocolFakeProviders
>>>
>>>
>>> providers = ProtocolFakeProviders.defaults()

"""

from tests._helpers.fakes.fake_providers import (
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
from tests._helpers.fakes.utilities import utcnow

__all__ = [
    "FakeGitHistoryProvider",
    "FakeTestReporter",
    "FakeTypeChecker",
    "ProtocolFakeProviders",
    "ProtocolFakeScipIndexer",
    "utcnow",
]
