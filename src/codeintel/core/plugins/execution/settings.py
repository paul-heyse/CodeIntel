"""Per-plugin execution settings for plugin executors.

This module provides the `PluginExecutionSettings` dataclass that captures
resolved per-plugin execution configuration including retry behavior,
timeouts, severity, and content hashes for cache invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.core.plugins.execution.policy import BaseExecutionPolicy

if TYPE_CHECKING:
    from codeintel.core.execution.retry import RetryPolicy
    from codeintel.core.plugins.types.protocol import PluginProtocol

PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]


@dataclass(frozen=True)
class PluginExecutionSettings:
    """Resolved execution settings for a single plugin.

    Contains all runtime configuration needed to execute a plugin,
    including retry behavior, timeouts, and content hashes for
    cache invalidation and skip detection.

    Attributes
    ----------
    name
        Plugin name.
    severity
        How failures should be handled.
    retry_policy
        Retry configuration for transient failures.
    timeout_ms
        Execution timeout in milliseconds, None for no timeout.
    fail_fast
        Whether to abort the run on first failure.
    input_hash
        Content hash of plugin inputs for cache invalidation.
    options_hash
        Content hash of plugin options.
    version_hash
        Plugin version hash for detecting code changes.

    Examples
    --------
    >>> from codeintel.core.execution.retry import RetryPolicy
    >>> settings = PluginExecutionSettings(
    ...     name="my_plugin",
    ...     severity="fatal",
    ...     retry_policy=RetryPolicy(max_attempts=3),
    ...     timeout_ms=30000,
    ...     fail_fast=True,
    ...     input_hash="abc123",
    ...     options_hash=None,
    ...     version_hash="v1.0",
    ... )
    >>> settings.name
    'my_plugin'
    """

    name: str
    severity: PluginSeverity
    retry_policy: RetryPolicy
    timeout_ms: int | None
    fail_fast: bool
    input_hash: str | None
    options_hash: str | None
    version_hash: str | None


def build_plugin_settings_from_policy(
    plugin: PluginProtocol,
    policy: BaseExecutionPolicy,
    *,
    input_hash: str | None = None,
    options_hash: str | None = None,
) -> PluginExecutionSettings:
    """Build execution settings for a plugin from a policy.

    Resolve per-plugin overrides for severity, retry, and timeout
    from the policy, falling back to defaults.

    Parameters
    ----------
    plugin
        Plugin to build settings for.
    policy
        Execution policy with defaults and overrides.
    input_hash
        Pre-computed input hash.
    options_hash
        Pre-computed options hash.

    Returns
    -------
    PluginExecutionSettings
        Resolved settings for the plugin.

    Examples
    --------
    >>> from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
    >>> policy = BaseExecutionPolicy(default_severity="soft_fail")
    >>> settings = build_plugin_settings_from_policy(plugin, policy)
    """
    name = plugin.metadata.name
    version_hash = plugin.metadata.version_hash

    return PluginExecutionSettings(
        name=name,
        severity=policy.get_severity(name),
        retry_policy=policy.get_retry_policy(name),
        timeout_ms=policy.get_timeout(name),
        fail_fast=policy.fail_fast,
        input_hash=input_hash,
        options_hash=options_hash,
        version_hash=version_hash,
    )


__all__ = [
    "PluginExecutionSettings",
    "PluginSeverity",
    "build_plugin_settings_from_policy",
]
