"""Plugin decorator utilities.

This package contains decorator-based patterns for plugin creation:

Functional Plugins
------------------
- BaseFunctionalPlugin: Base class for function-based plugins

Meta Options
------------
- BasePluginMetaOptions: Dataclass for plugin configuration
- BasePluginMetaOptionsInput: TypedDict for input parsing

Factory Functions
-----------------
- make_plugin_instance: Create plugin from function and options
"""

from __future__ import annotations

from codeintel.core.plugins.decorators.functional import BaseFunctionalPlugin
from codeintel.core.plugins.decorators.meta import (
    BasePluginMetaOptions,
    BasePluginMetaOptionsInput,
)
from codeintel.core.plugins.decorators.step import make_plugin_instance

__all__ = [
    "BaseFunctionalPlugin",
    "BasePluginMetaOptions",
    "BasePluginMetaOptionsInput",
    "make_plugin_instance",
]
