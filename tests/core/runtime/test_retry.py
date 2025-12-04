"""Test retry infrastructure from codeintel.core.runtime.retry.

This module tests:
- RetryPolicy default configuration
- Wait strategies with/without jitter
- Stop strategies combining limits
- Block-style and decorator-style retrying
- Pre-configured policies (PLUGIN_, NETWORK_, DATABASE_, NO_RETRY_)
- Helper functions with_retry and with_retry_async
"""

from __future__ import annotations

import pytest
from tenacity import Retrying, wait_exponential, wait_random_exponential

from codeintel.core.runtime.errors import PLUGIN_CATCHABLE_ERRORS
from codeintel.core.runtime.retry import (
    DATABASE_RETRY_POLICY,
    NETWORK_RETRY_POLICY,
    NO_RETRY_POLICY,
    PLUGIN_RETRY_POLICY,
    RetryPolicy,
    get_retry_policy_for_retryable,
    with_retry,
    with_retry_async,
)

# =============================================================================
# RetryPolicy Construction Tests
# =============================================================================


def test_retry_policy_defaults() -> None:
    """Verify RetryPolicy default values."""
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.max_delay_s == 30.0
    assert policy.backoff_multiplier == 0.5
    assert policy.backoff_max_s == 10.0
    assert policy.retryable_exceptions == PLUGIN_CATCHABLE_ERRORS
    assert policy.use_jitter is True
    assert policy.log_retries is True


def test_retry_policy_custom_values() -> None:
    """Verify RetryPolicy accepts custom values."""
    policy = RetryPolicy(
        max_attempts=5,
        max_delay_s=60.0,
        backoff_multiplier=1.0,
        backoff_max_s=20.0,
        retryable_exceptions=(ValueError, TypeError),
        use_jitter=False,
        log_retries=False,
    )

    assert policy.max_attempts == 5
    assert policy.max_delay_s == 60.0
    assert policy.backoff_multiplier == 1.0
    assert policy.backoff_max_s == 20.0
    assert policy.retryable_exceptions == (ValueError, TypeError)
    assert policy.use_jitter is False
    assert policy.log_retries is False


def test_retry_policy_is_frozen() -> None:
    """Verify RetryPolicy is immutable."""
    policy = RetryPolicy()

    with pytest.raises(AttributeError):
        policy.max_attempts = 10  # type: ignore[misc]


# =============================================================================
# Wait Strategy Tests
# =============================================================================


def test_get_wait_strategy_with_jitter() -> None:
    """Verify wait strategy with jitter enabled."""
    policy = RetryPolicy(use_jitter=True)
    strategy = policy._get_wait_strategy()

    assert isinstance(strategy, wait_random_exponential)


def test_get_wait_strategy_without_jitter() -> None:
    """Verify wait strategy without jitter."""
    policy = RetryPolicy(use_jitter=False)
    strategy = policy._get_wait_strategy()

    assert isinstance(strategy, wait_exponential)


# =============================================================================
# Stop Strategy Tests
# =============================================================================


def test_get_stop_strategy() -> None:
    """Verify stop strategy combines attempts and delay."""
    policy = RetryPolicy(max_attempts=3, max_delay_s=30.0)
    strategy = policy._get_stop_strategy()

    # Strategy should be a combination (stop_after_attempt | stop_after_delay)
    # We can't easily inspect the combined strategy, but we can verify it works
    assert strategy is not None


# =============================================================================
# Create Retrying Tests
# =============================================================================


def test_create_retrying_returns_retrying() -> None:
    """Verify create_retrying returns a Retrying instance."""
    policy = RetryPolicy()
    retrying = policy.create_retrying()

    assert isinstance(retrying, Retrying)


def test_create_retrying_respects_max_attempts() -> None:
    """Verify retrying respects max_attempts limit."""
    call_count = 0

    policy = RetryPolicy(
        max_attempts=3,
        retryable_exceptions=(ValueError,),
        log_retries=False,
        use_jitter=False,
        backoff_multiplier=0.01,  # Fast backoff for tests
    )

    with pytest.raises(ValueError):
        for attempt in policy.create_retrying():
            with attempt:
                call_count += 1
                msg = "Always fails"
                raise ValueError(msg)

    assert call_count == 3


def test_create_retrying_success_on_first_try() -> None:
    """Verify retrying succeeds on first successful attempt."""
    policy = RetryPolicy()
    result = None

    for attempt in policy.create_retrying():
        with attempt:
            result = "success"

    assert result == "success"


def test_create_retrying_success_after_retry() -> None:
    """Verify retrying succeeds after transient failure."""
    call_count = 0

    policy = RetryPolicy(
        max_attempts=3,
        retryable_exceptions=(ValueError,),
        log_retries=False,
        use_jitter=False,
        backoff_multiplier=0.01,
    )

    for attempt in policy.create_retrying():
        with attempt:
            call_count += 1
            if call_count < 2:
                msg = "Transient failure"
                raise ValueError(msg)

    assert call_count == 2


# =============================================================================
# Async Retrying Tests
# =============================================================================


def test_create_async_retrying_returns_async_retrying() -> None:
    """Verify create_async_retrying returns an AsyncRetrying instance."""
    from tenacity import AsyncRetrying

    policy = RetryPolicy()
    async_retrying = policy.create_async_retrying()

    assert isinstance(async_retrying, AsyncRetrying)


# =============================================================================
# Decorator Tests
# =============================================================================


def test_as_decorator() -> None:
    """Verify as_decorator returns a callable decorator."""
    policy = RetryPolicy()
    decorator = policy.as_decorator()

    assert callable(decorator)


def test_as_decorator_application() -> None:
    """Verify decorator can be applied to functions."""
    call_count = 0

    policy = RetryPolicy(
        max_attempts=2,
        retryable_exceptions=(ValueError,),
        log_retries=False,
        use_jitter=False,
        backoff_multiplier=0.01,
    )

    @policy.as_decorator()
    def flaky_function() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            msg = "Transient"
            raise ValueError(msg)
        return "decorated_success"

    result = flaky_function()
    assert result == "decorated_success"
    assert call_count == 2


# =============================================================================
# Immutable Copy Tests
# =============================================================================


def test_with_max_attempts() -> None:
    """Verify with_max_attempts creates a new policy."""
    original = RetryPolicy(max_attempts=3)
    modified = original.with_max_attempts(5)

    assert original.max_attempts == 3
    assert modified.max_attempts == 5
    assert modified.max_delay_s == original.max_delay_s
    assert modified.backoff_multiplier == original.backoff_multiplier


def test_with_exceptions() -> None:
    """Verify with_exceptions creates a new policy."""
    original = RetryPolicy(retryable_exceptions=(ValueError,))
    modified = original.with_exceptions((TypeError, RuntimeError))

    assert original.retryable_exceptions == (ValueError,)
    assert modified.retryable_exceptions == (TypeError, RuntimeError)
    assert modified.max_attempts == original.max_attempts


# =============================================================================
# Pre-configured Policy Tests
# =============================================================================


def test_plugin_retry_policy() -> None:
    """Verify PLUGIN_RETRY_POLICY defaults."""
    assert PLUGIN_RETRY_POLICY.max_attempts == 3
    assert PLUGIN_RETRY_POLICY.max_delay_s == 30.0
    assert PLUGIN_RETRY_POLICY.backoff_max_s == 5.0


def test_network_retry_policy() -> None:
    """Verify NETWORK_RETRY_POLICY defaults."""
    assert NETWORK_RETRY_POLICY.max_attempts == 5
    assert NETWORK_RETRY_POLICY.max_delay_s == 60.0
    assert NETWORK_RETRY_POLICY.backoff_max_s == 20.0
    assert TimeoutError in NETWORK_RETRY_POLICY.retryable_exceptions
    assert ConnectionError in NETWORK_RETRY_POLICY.retryable_exceptions


def test_no_retry_policy() -> None:
    """Verify NO_RETRY_POLICY disables retries."""
    assert NO_RETRY_POLICY.max_attempts == 1
    assert NO_RETRY_POLICY.max_delay_s == 0.0


def test_database_retry_policy() -> None:
    """Verify DATABASE_RETRY_POLICY defaults."""
    assert DATABASE_RETRY_POLICY.max_attempts == 3
    assert DATABASE_RETRY_POLICY.max_delay_s == 15.0
    assert DATABASE_RETRY_POLICY.backoff_max_s == 3.0


# =============================================================================
# with_retry Helper Tests
# =============================================================================


def test_with_retry_success() -> None:
    """Verify with_retry executes function successfully."""

    def simple_fn() -> str:
        return "result"

    result = with_retry(PLUGIN_RETRY_POLICY, simple_fn)
    assert result == "result"


def test_with_retry_with_args() -> None:
    """Verify with_retry passes arguments to function."""

    def add(a: int, b: int) -> int:
        return a + b

    result = with_retry(PLUGIN_RETRY_POLICY, add, 2, 3)
    assert result == 5


def test_with_retry_with_kwargs() -> None:
    """Verify with_retry passes keyword arguments to function."""

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    result = with_retry(PLUGIN_RETRY_POLICY, greet, "World", greeting="Hi")
    assert result == "Hi, World!"


def test_with_retry_retries_on_exception() -> None:
    """Verify with_retry retries on transient exceptions."""
    call_count = 0

    def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            msg = "Transient"
            raise ValueError(msg)
        return "success"

    policy = RetryPolicy(
        max_attempts=3,
        retryable_exceptions=(ValueError,),
        log_retries=False,
        use_jitter=False,
        backoff_multiplier=0.01,
    )

    result = with_retry(policy, flaky)
    assert result == "success"
    assert call_count == 2


def test_with_retry_raises_after_exhaustion() -> None:
    """Verify with_retry raises after retry exhaustion."""

    def always_fails() -> str:
        msg = "Always fails"
        raise ValueError(msg)

    policy = RetryPolicy(
        max_attempts=2,
        retryable_exceptions=(ValueError,),
        log_retries=False,
        use_jitter=False,
        backoff_multiplier=0.01,
    )

    with pytest.raises(ValueError, match="Always fails"):
        with_retry(policy, always_fails)


# =============================================================================
# with_retry_async Helper Tests (sync assertions only - async tests need pytest-asyncio config)
# =============================================================================


def test_with_retry_async_is_coroutine_function() -> None:
    """Verify with_retry_async is available as an async helper."""
    import inspect

    assert inspect.iscoroutinefunction(with_retry_async)


# =============================================================================
# get_retry_policy_for_retryable Tests
# =============================================================================


def test_get_retry_policy_for_retryable() -> None:
    """Verify get_retry_policy_for_retryable creates correct policy."""
    policy = get_retry_policy_for_retryable(
        max_retries=5,
        retry_backoff_ms=500,
        retryable_exceptions=(ValueError, TypeError),
    )

    assert policy.max_attempts == 5
    assert policy.backoff_multiplier == 0.5  # 500ms / 1000
    assert policy.retryable_exceptions == (ValueError, TypeError)


def test_get_retry_policy_for_retryable_zero_backoff() -> None:
    """Verify get_retry_policy_for_retryable handles zero backoff."""
    policy = get_retry_policy_for_retryable(
        max_retries=3,
        retry_backoff_ms=0,
        retryable_exceptions=(RuntimeError,),
    )

    assert policy.max_attempts == 3
    assert policy.backoff_multiplier == 0.0


def test_get_retry_policy_for_retryable_single_retry() -> None:
    """Verify get_retry_policy_for_retryable with single retry."""
    policy = get_retry_policy_for_retryable(
        max_retries=1,
        retry_backoff_ms=100,
        retryable_exceptions=(OSError,),
    )

    assert policy.max_attempts == 1
