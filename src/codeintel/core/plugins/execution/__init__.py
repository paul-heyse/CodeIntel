"""Plugin execution infrastructure.

This package contains execution context and support for plugin execution:

Context Types
-------------
- PluginExecutionContext: Runtime context for plugin execution
- PluginScratch: Inter-plugin data sharing store
- ConfigProvider: Typed configuration access
- PluginExecutionContextBuilder: Builder pattern for contexts

Options
-------
- PluginOptionsResolver: Resolve plugin options from config sources
- ConfigSource: Protocol for configuration sources
- EmptyConfigSource: Default empty config source
- ProfiledConfigSource: Config source backed by ExecutionProfile

Profiles
--------
- ExecutionProfile: Profile defining plugin behavior
- get_profile: Get profile by name
- register_profile: Register a custom profile
- FULL_PROFILE, FAST_PROFILE, CI_PROFILE: Built-in profiles
"""

from __future__ import annotations

from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginConfigBundle,
    PluginOptionsResolver,
    ProfiledConfigSource,
)
from codeintel.core.plugins.execution.profiles import (
    CI_PROFILE,
    FAST_PROFILE,
    FULL_PROFILE,
    ExecutionProfile,
    get_profile,
    list_profiles,
    register_profile,
)

__all__ = [
    "CI_PROFILE",
    "FAST_PROFILE",
    "FULL_PROFILE",
    "ConfigProvider",
    "ConfigSource",
    "EmptyConfigSource",
    "ExecutionProfile",
    "PluginConfigBundle",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginOptionsResolver",
    "PluginScratch",
    "ProfiledConfigSource",
    "get_profile",
    "list_profiles",
    "register_profile",
]
