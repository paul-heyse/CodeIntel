"""Test plugin traits from codeintel.core.plugins.traits.

This module tests:
- IsolatedPlugin, CacheAwarePlugin, RetryablePlugin, ProgressReportingPlugin, IncrementalPlugin protocols
- CacheAwareMixin, RetryableMixin, ProgressReportingMixin mixins
- WithDependencyData mixin
- Detection utilities (is_isolated, is_cache_aware, is_retryable, etc.)
- get_retry_policy function
"""

from __future__ import annotations

from typing import Literal

from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.plugins.traits import (
    CacheAwareMixin,
    CacheAwarePlugin,
    IncrementalPlugin,
    IsolatedPlugin,
    ProgressReportingMixin,
    ProgressReportingPlugin,
    RetryableMixin,
    RetryablePlugin,
    WithDependencyData,
    get_retry_policy,
    is_cache_aware,
    is_incremental,
    is_isolated,
    is_progress_reporting,
    is_retryable,
)
from codeintel.core.execution.retry import PLUGIN_RETRY_POLICY, RetryPolicy

_default_retryable = RetryableMixin()
DEFAULT_MAX_RETRIES = _default_retryable.max_retries
DEFAULT_RETRY_BACKOFF_MS = _default_retryable.retry_backoff_ms
CUSTOM_MAX_RETRIES = 10
CUSTOM_RETRY_BACKOFF_MS = 5000
POLICY_MAX_ATTEMPTS = 7
POLICY_BACKOFF_MS = 3000
MIXIN_POLICY_ATTEMPTS = 8
RETRYABLE_PLUGIN_ATTEMPTS = 5
PROGRESS_EVENTS_EXPECTED = 2

# =============================================================================
# Protocol Implementation Classes for Testing
# =============================================================================


class TestIsolatedPlugin:
    """Test implementation of IsolatedPlugin."""

    @property
    def isolation_kind(self) -> Literal["process", "thread", "none"]:
        """Return process isolation."""
        return "process"


class TestCacheAwarePlugin:
    """Test implementation of CacheAwarePlugin."""

    @property
    def cache_populates(self) -> tuple[str, ...]:
        """Return cache keys populated."""
        return ("metrics",)

    @property
    def cache_consumes(self) -> tuple[str, ...]:
        """Return cache keys consumed."""
        return ("goids",)


class TestRetryablePlugin:
    """Test implementation of RetryablePlugin."""

    @property
    def retryable_exceptions(self) -> tuple[type[Exception], ...]:
        """Return retryable exceptions."""
        return (TimeoutError, ConnectionError)

    @property
    def max_retries(self) -> int:
        """Return max retries."""
        return RETRYABLE_PLUGIN_ATTEMPTS

    @property
    def retry_backoff_ms(self) -> int:
        """Return backoff ms."""
        return 2000


class TestProgressReportingPlugin:
    """Test implementation of ProgressReportingPlugin."""

    def __init__(self) -> None:
        """Initialize test plugin."""
        self.callback: object = None

    def set_progress_callback(
        self,
        callback: object,
    ) -> None:
        """Set progress callback."""
        self.callback = callback


class TestIncrementalPlugin:
    """Test implementation of IncrementalPlugin."""

    def __init__(self) -> None:
        """Initialize incremental plugin."""
        self._hash_value = "abc123"
        self.last_context: object | None = None

    def compute_input_hash(self, ctx: object) -> str:
        """Compute hash.

        Returns
        -------
        str
            Fixed hash value.
        """
        self.last_context = ctx
        return self._hash_value

    def is_unchanged(self, ctx: object, prior_hash: str | None) -> bool:
        """Check if unchanged.

        Returns
        -------
        bool
            True if prior hash matches expected.
        """
        self.last_context = ctx
        return prior_hash == self._hash_value


class PlainPlugin:
    """Plugin without any traits."""


# =============================================================================
# Protocol Detection Tests
# =============================================================================


def test_is_isolated_true() -> None:
    """Verify is_isolated returns True for IsolatedPlugin."""
    plugin = TestIsolatedPlugin()
    assert is_isolated(plugin) is True


def test_is_isolated_false() -> None:
    """Verify is_isolated returns False for non-isolated plugin."""
    plugin = PlainPlugin()
    assert is_isolated(plugin) is False


def test_is_cache_aware_true() -> None:
    """Verify is_cache_aware returns True for CacheAwarePlugin."""
    plugin = TestCacheAwarePlugin()
    assert is_cache_aware(plugin) is True


def test_is_cache_aware_false() -> None:
    """Verify is_cache_aware returns False for non-cache plugin."""
    plugin = PlainPlugin()
    assert is_cache_aware(plugin) is False


def test_is_retryable_true() -> None:
    """Verify is_retryable returns True for RetryablePlugin."""
    plugin = TestRetryablePlugin()
    assert is_retryable(plugin) is True


def test_is_retryable_false() -> None:
    """Verify is_retryable returns False for non-retryable plugin."""
    plugin = PlainPlugin()
    assert is_retryable(plugin) is False


def test_is_progress_reporting_true() -> None:
    """Verify is_progress_reporting returns True for ProgressReportingPlugin."""
    plugin = TestProgressReportingPlugin()
    assert is_progress_reporting(plugin) is True


def test_is_progress_reporting_false() -> None:
    """Verify is_progress_reporting returns False for non-progress plugin."""
    plugin = PlainPlugin()
    assert is_progress_reporting(plugin) is False


def test_is_incremental_true() -> None:
    """Verify is_incremental returns True for IncrementalPlugin."""
    plugin = TestIncrementalPlugin()
    assert is_incremental(plugin) is True


def test_is_incremental_false() -> None:
    """Verify is_incremental returns False for non-incremental plugin."""
    plugin = PlainPlugin()
    assert is_incremental(plugin) is False


# =============================================================================
# CacheAwareMixin Tests
# =============================================================================


def test_cache_aware_mixin_defaults() -> None:
    """Verify CacheAwareMixin has empty defaults."""
    mixin = CacheAwareMixin()
    assert mixin.cache_populates == ()
    assert mixin.cache_consumes == ()


def test_cache_aware_mixin_custom_values() -> None:
    """Verify CacheAwareMixin respects class attributes."""

    class CustomCacheMixin(CacheAwareMixin):
        _cache_populates = ("data1", "data2")
        _cache_consumes = ("upstream",)

    mixin = CustomCacheMixin()
    assert mixin.cache_populates == ("data1", "data2")
    assert mixin.cache_consumes == ("upstream",)


def test_cache_aware_mixin_is_protocol_compliant() -> None:
    """Verify CacheAwareMixin satisfies CacheAwarePlugin protocol."""

    class CachingPlugin(CacheAwareMixin):
        _cache_populates = ("result",)
        _cache_consumes = ("input",)

    plugin = CachingPlugin()
    assert isinstance(plugin, CacheAwarePlugin)


# =============================================================================
# RetryableMixin Tests
# =============================================================================


def test_retryable_mixin_defaults() -> None:
    """Verify RetryableMixin has sensible defaults."""
    mixin = RetryableMixin()
    assert mixin.retryable_exceptions == (RuntimeError, ValueError, OSError)
    assert mixin.max_retries == DEFAULT_MAX_RETRIES
    assert mixin.retry_backoff_ms == DEFAULT_RETRY_BACKOFF_MS


def test_retryable_mixin_custom_values() -> None:
    """Verify RetryableMixin respects class attributes."""

    class CustomRetryMixin(RetryableMixin):
        _retryable_exceptions = (TimeoutError,)
        _max_retries = CUSTOM_MAX_RETRIES
        _retry_backoff_ms = CUSTOM_RETRY_BACKOFF_MS

    mixin = CustomRetryMixin()
    assert mixin.retryable_exceptions == (TimeoutError,)
    assert mixin.max_retries == CUSTOM_MAX_RETRIES
    assert mixin.retry_backoff_ms == CUSTOM_RETRY_BACKOFF_MS


def test_retryable_mixin_get_retry_policy() -> None:
    """Verify RetryableMixin.get_retry_policy returns configured policy."""

    class RetryPlugin(RetryableMixin):
        _max_retries = POLICY_MAX_ATTEMPTS
        _retry_backoff_ms = POLICY_BACKOFF_MS
        _retryable_exceptions = (ValueError,)

    mixin = RetryPlugin()
    policy = mixin.get_retry_policy()

    assert isinstance(policy, RetryPolicy)
    assert policy.max_attempts == POLICY_MAX_ATTEMPTS
    assert policy.retryable_exceptions == (ValueError,)


def test_retryable_mixin_is_protocol_compliant() -> None:
    """Verify RetryableMixin satisfies RetryablePlugin protocol."""
    mixin = RetryableMixin()
    assert isinstance(mixin, RetryablePlugin)


# =============================================================================
# ProgressReportingMixin Tests
# =============================================================================


def test_progress_reporting_mixin_no_callback() -> None:
    """Verify report_progress is no-op without callback."""
    mixin = ProgressReportingMixin()
    # Should not raise
    mixin.report_progress(0.5, "Working")


def test_progress_reporting_mixin_with_callback() -> None:
    """Verify report_progress calls callback."""
    mixin = ProgressReportingMixin()
    received: list[tuple[float, str]] = []

    def callback(progress: float, message: str) -> None:
        received.append((progress, message))

    mixin.set_progress_callback(callback)
    mixin.report_progress(0.5, "Halfway")
    mixin.report_progress(1.0, "Done")

    assert len(received) == PROGRESS_EVENTS_EXPECTED
    assert received[0] == (0.5, "Halfway")
    assert received[1] == (1.0, "Done")


def test_progress_reporting_mixin_is_protocol_compliant() -> None:
    """Verify ProgressReportingMixin satisfies ProgressReportingPlugin protocol."""
    mixin = ProgressReportingMixin()
    assert isinstance(mixin, ProgressReportingPlugin)


# =============================================================================
# WithDependencyData Tests
# =============================================================================


class MockContext:
    """Mock context with scratch store that satisfies ScratchContext protocol."""

    def __init__(self) -> None:
        """Initialize mock context."""
        self._scratch = PluginScratch()

    @property
    def scratch(self) -> PluginScratch:
        """Return scratch store."""
        return self._scratch


def test_with_dependency_data_get() -> None:
    """Verify get_dependency_data retrieves data."""
    ctx = MockContext()
    ctx.scratch.declare("test_key", {"value": 42})

    result = WithDependencyData.get_dependency_data(ctx, "test_key")

    assert result == {"value": 42}


def test_with_dependency_data_get_default() -> None:
    """Verify get_dependency_data returns default for missing key."""
    ctx = MockContext()

    result = WithDependencyData.get_dependency_data(ctx, "missing", "default")

    assert result == "default"


def test_with_dependency_data_set() -> None:
    """Verify set_dependency_data stores data."""
    ctx = MockContext()

    WithDependencyData.set_dependency_data(ctx, "output", {"result": 100})

    assert ctx.scratch.consume("output") == {"result": 100}


# =============================================================================
# get_retry_policy Function Tests
# =============================================================================


def test_get_retry_policy_with_mixin() -> None:
    """Verify get_retry_policy uses mixin's get_retry_policy method."""

    class MixinPlugin(RetryableMixin):
        _max_retries = MIXIN_POLICY_ATTEMPTS

    plugin = MixinPlugin()
    policy = get_retry_policy(plugin)

    assert isinstance(policy, RetryPolicy)
    assert policy.max_attempts == MIXIN_POLICY_ATTEMPTS


def test_get_retry_policy_with_protocol() -> None:
    """Verify get_retry_policy uses RetryablePlugin attributes."""
    plugin = TestRetryablePlugin()
    policy = get_retry_policy(plugin)

    assert isinstance(policy, RetryPolicy)
    assert policy.max_attempts == RETRYABLE_PLUGIN_ATTEMPTS
    assert policy.retryable_exceptions == (TimeoutError, ConnectionError)


def test_get_retry_policy_default() -> None:
    """Verify get_retry_policy returns default for plain plugin."""
    plugin = PlainPlugin()
    policy = get_retry_policy(plugin)

    assert policy is PLUGIN_RETRY_POLICY


def test_get_retry_policy_method_returns_non_policy() -> None:
    """Verify get_retry_policy falls back when method returns non-policy."""

    class BadPolicyPlugin(RetryableMixin):
        def get_retry_policy(self) -> str:  # type: ignore[override]
            return f"{self.__class__.__name__} policy"

    plugin = BadPolicyPlugin()
    policy = get_retry_policy(plugin)

    # Should fall back to constructing from mixin attributes
    assert isinstance(policy, RetryPolicy)


# =============================================================================
# Protocol Runtime Checkable Tests
# =============================================================================


def test_isolated_plugin_is_runtime_checkable() -> None:
    """Verify IsolatedPlugin is runtime_checkable."""
    assert isinstance(TestIsolatedPlugin(), IsolatedPlugin)


def test_cache_aware_plugin_is_runtime_checkable() -> None:
    """Verify CacheAwarePlugin is runtime_checkable."""
    assert isinstance(TestCacheAwarePlugin(), CacheAwarePlugin)


def test_retryable_plugin_is_runtime_checkable() -> None:
    """Verify RetryablePlugin is runtime_checkable."""
    assert isinstance(TestRetryablePlugin(), RetryablePlugin)


def test_progress_reporting_plugin_is_runtime_checkable() -> None:
    """Verify ProgressReportingPlugin is runtime_checkable."""
    assert isinstance(TestProgressReportingPlugin(), ProgressReportingPlugin)


def test_incremental_plugin_is_runtime_checkable() -> None:
    """Verify IncrementalPlugin is runtime_checkable."""
    assert isinstance(TestIncrementalPlugin(), IncrementalPlugin)
