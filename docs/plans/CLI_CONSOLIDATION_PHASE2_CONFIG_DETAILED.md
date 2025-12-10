# CLI Consolidation Phase 2: Config Integration — Detailed Implementation Plan

> **Status**: Ready for Implementation  
> **Depends On**: Phase 1 (Foundation Layer) ✅ Complete  
> **Enables**: Phase 3 (Handler Migration)  
> **Risk Level**: Low (mostly removals and wiring changes)  
> **Reference**: [CLI_CONSOLIDATION_ARCHITECTURE.md](../CLI_CONSOLIDATION_ARCHITECTURE.md)

## Overview

This document provides step-by-step implementation instructions for Phase 2 of the CLI consolidation. Phase 2 integrates `ConfigService` (created in Phase 1) into the Cyclopts application root, removing duplicate configuration loading code.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Implementation Order](#2-implementation-order)
3. [Step 1: Enhance ConfigService for TOML Support](#3-step-1-enhance-configservice-for-toml-support)
4. [Step 2: Update make_root_app()](#4-step-2-update-make_root_app)
5. [Step 3: Remove Duplicates from cyclopts_common.py](#5-step-3-remove-duplicates-from-cyclopts_commonpy)
6. [Step 4: Refactor cyclopts_config.py Commands](#6-step-4-refactor-cyclopts_configpy-commands)
7. [Step 5: Add Deprecation Warnings](#7-step-5-add-deprecation-warnings)
8. [Step 6: Test Implementation](#8-step-6-test-implementation)
9. [Verification Checklist](#9-verification-checklist)

---

## 1. Current State Analysis

### 1.1 Phase 1 Deliverables (Complete)

| Component | Location | Status |
|-----------|----------|--------|
| `ConfigService` | `cli/config/service.py` | ✅ Implemented |
| `CONFIG_ENV_PREFIX` | `cli/config/service.py` | ✅ Canonical location |
| `RuntimeParams` | `cli/resolution/params.py` | ✅ Implemented |
| `EnhancedHandlerContext` | `cli/handlers/protocol.py` | ✅ Implemented |
| `UnifiedRenderer` | `cli/rendering/service.py` | ✅ Implemented |

### 1.2 Duplicate Code Inventory (To Remove)

#### Constants

| Location | Code | Action |
|----------|------|--------|
| `cyclopts_common.py:41` | `CONFIG_ENV_PREFIX = "CODEINTEL_"` | **DELETE** |
| `cyclopts_common.py:42` | `CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"` | **DELETE** |
| `cyclopts_common.py:43` | `DEFAULT_CONFIG_PATH = Path("codeintel.toml")` | **DELETE** |
| `cyclopts_common.py:45` | `_ENV_CONFIG = cyclopts_config.Env(...)` | **DELETE** |
| `cyclopts_config.py:24` | `CONFIG_ENV_PREFIX = "CODEINTEL_"` | **KEEP** (local to config commands) |
| `cyclopts_config.py:25` | `CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"` | **KEEP** (local to config commands) |
| `cyclopts_config.py:26` | `DEFAULT_CONFIG_PATH = Path("codeintel.toml")` | **KEEP** (local to config commands) |

#### Functions

| Location | Function | Action |
|----------|----------|--------|
| `cyclopts_common.py:48-58` | `_resolve_config_path()` | **DELETE** |
| `cyclopts_common.py:60-74` | `_optional_toml_config()` | **DELETE** |
| `cyclopts_common.py:77-94` | `make_root_app()` | **REFACTOR** to use ConfigService |

### 1.3 Key Integration Point

The current `make_root_app()` function uses:

```python
# Current (cyclopts_common.py:77-94)
def make_root_app() -> App:
    return App(
        name="codeintel",
        help="...",
        config=[_optional_toml_config, _ENV_CONFIG],  # OLD: Remove this
        ...
    )
```

Target state:

```python
# Target: Use ConfigService
def make_root_app() -> App:
    return App(
        name="codeintel",
        help="...",
        config=ConfigService.get_cyclopts_config_chain(),  # NEW: Use ConfigService
        ...
    )
```

---

## 2. Implementation Order

Create/modify files in this order to minimize breakage:

```
1. cli/config/service.py       # Enhance with TOML and Cyclopts integration
2. cli/cyclopts_common.py      # Update make_root_app(), remove duplicates
3. cyclopts_config.py          # Refactor commands to use ConfigService where appropriate
4. Tests                       # Add regression and deprecation tests
```

---

## 3. Step 1: Enhance ConfigService for TOML Support

The current `ConfigService` returns an empty config chain. We need to enhance it to support TOML files and Cyclopts integration for the `make_root_app()` migration.

### 3.1 Update `cli/config/service.py`

Add TOML config path support and Cyclopts integration:

```python
"""Unified configuration service.

This module provides ConfigService, the single source of truth for all
CLI configuration loading, validation, and access.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from cyclopts import config as cyclopts_config

from codeintel.cli.config.loader import apply_overrides, load_config
from codeintel.cli.config.model import CliConfig

if TYPE_CHECKING:
    from cyclopts import App


# Environment variable prefix (canonical location)
CONFIG_ENV_PREFIX = "CODEINTEL_"

# Environment variable for explicit config path
CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"

# TOML config file search paths
TOML_CONFIG_PATHS = [
    Path("codeintel.toml"),
    Path.home() / ".codeintel" / "config.toml",
]


def _resolve_toml_config_path() -> Path | None:
    """Return the configured TOML path if it exists.

    Check environment override first, then search default locations.

    Returns
    -------
    Path | None
        Path to the config file if found, None otherwise.
    """
    # Check environment override
    env_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Search default locations
    for path in TOML_CONFIG_PATHS:
        if path.exists():
            return path

    return None


def _make_optional_toml_config() -> Callable[[App, tuple[str, ...], Any], Any]:
    """Create TOML config callable for Cyclopts.

    Returns
    -------
    Callable
        Config callable that applies TOML config if present.
    """
    def _apply_toml(apps: object, commands: tuple[str, ...], arguments: object) -> object:
        """Apply TOML config if present; otherwise return unchanged."""
        path = _resolve_toml_config_path()
        if path is None:
            return arguments

        toml_loader = cast("Any", cyclopts_config.Toml(str(path)))
        app_arg = cast("App", apps)
        args_arg = cast("Any", arguments)
        return toml_loader(app_arg, commands, args_arg)

    return _apply_toml


@dataclass(frozen=True)
class ConfigService:
    """Unified configuration service.

    Precedence (highest to lowest):

    1. CLI flags (explicit overrides)
    2. Environment variables (CODEINTEL_*)
    3. Config file (codeintel.toml or ~/.codeintel/config.yaml)
    4. Built-in defaults from CliConfig

    Parameters
    ----------
    config
        The resolved, validated configuration.
    sources
        Ordered list of sources that contributed to the config.

    Examples
    --------
    >>> service = ConfigService.load(validate=False)
    >>> service.config.output_format
    'text'
    >>> "defaults" in service.sources
    True
    """

    config: CliConfig
    sources: tuple[str, ...]

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
        *,
        env_prefix: str = CONFIG_ENV_PREFIX,
        validate: bool = True,
    ) -> ConfigService:
        """Load configuration from all sources with precedence.

        Parameters
        ----------
        config_path
            Explicit config file path. If None, searches default locations.
        cli_overrides
            Overrides from CLI flags (highest precedence).
        env_prefix
            Environment variable prefix.
        validate
            If True, validate config and raise ConfigLoadError on failure.

        Returns
        -------
        ConfigService
            Service with loaded configuration.

        Examples
        --------
        >>> service = ConfigService.load(validate=False)
        >>> service.config.color
        True
        """
        # Use the existing load_config with its precedence
        config = load_config(
            config_file=config_path,
            env_prefix=env_prefix,
            cli_overrides=cli_overrides,
            validate=validate,
        )

        return cls(
            config=config,
            sources=tuple(config.config_sources),
        )

    @staticmethod
    def get_cyclopts_config_chain() -> list[Callable[[App, tuple[str, ...], Any], Any]]:
        """Return Cyclopts-compatible config callables.

        Integrate with Cyclopts' config parameter while maintaining
        our unified precedence. The returned chain:

        1. Applies TOML config if present (codeintel.toml)
        2. Applies environment variable overrides (CODEINTEL_*)

        Returns
        -------
        list
            Config callables for Cyclopts App.config parameter.

        Examples
        --------
        >>> chain = ConfigService.get_cyclopts_config_chain()
        >>> len(chain) == 2
        True
        """
        return [
            _make_optional_toml_config(),
            cyclopts_config.Env(CONFIG_ENV_PREFIX),
        ]

    @staticmethod
    def get_toml_config_path() -> Path | None:
        """Get the resolved TOML config path if it exists.

        Returns
        -------
        Path | None
            Path to TOML config file or None.

        Examples
        --------
        >>> path = ConfigService.get_toml_config_path()
        >>> path is None or path.suffix == ".toml"
        True
        """
        return _resolve_toml_config_path()

    def with_overrides(self, **overrides: object) -> ConfigService:
        """Create new service with overrides applied.

        Useful for testing or command-specific modifications.

        Parameters
        ----------
        **overrides
            Field overrides to apply.

        Returns
        -------
        ConfigService
            New service with overrides applied.

        Examples
        --------
        >>> service = ConfigService.load(validate=False)
        >>> modified = service.with_overrides(color=False)
        >>> modified.config.color
        False
        """
        new_config = apply_overrides(self.config, overrides)
        return ConfigService(
            config=new_config,
            sources=(*self.sources, "overrides"),
        )


__all__ = [
    "CONFIG_ENV_PREFIX",
    "CONFIG_PATH_ENV_VAR",
    "ConfigService",
    "TOML_CONFIG_PATHS",
]
```

### 3.2 Update `cli/config/__init__.py`

Add new exports:

```python
# Add to imports
from codeintel.cli.config.service import (
    CONFIG_ENV_PREFIX,
    CONFIG_PATH_ENV_VAR,
    ConfigService,
    TOML_CONFIG_PATHS,
)

# Add to __all__
__all__ = [
    # ... existing ...
    "CONFIG_ENV_PREFIX",
    "CONFIG_PATH_ENV_VAR",
    "ConfigService",
    "TOML_CONFIG_PATHS",
]
```

---

## 4. Step 2: Update make_root_app()

### 4.1 Modify `cli/cyclopts_common.py`

Replace the old config loading with `ConfigService`:

```python
"""Shared Cyclopts primitives and runtime helpers for the CodeIntel CLI.

Configuration precedence
------------------------
CLI flags override environment variables (``CODEINTEL_*``), which override the
optional TOML config file (``codeintel.toml`` or ``CODEINTEL_CONFIG_PATH``),
which finally fall back to defaults in function signatures.

Execution model
---------------
The root :class:`cyclopts.App` is configured with ``result_action`` set to
``["call_if_callable", "return_value"]`` so commands can be embedded in tests
or other orchestrators without forcing ``sys.exit``. Commands implemented as
dataclasses with ``__call__`` will run naturally under this policy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter

from codeintel.cli.cli_types import BackendFlags, OutputFormat
from codeintel.cli.common_handlers import RuntimeCliOptions, build_config_from_options
from codeintel.cli.config import ConfigService  # NEW: Import ConfigService
from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    ProjectRuntime,
    StorageProjectConfig,
    build_project_runtime,
)
from codeintel.config.models import CliPathsInput
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, open_gateway

# REMOVED: CONFIG_ENV_PREFIX, CONFIG_PATH_ENV_VAR, DEFAULT_CONFIG_PATH
# REMOVED: _ENV_CONFIG, _resolve_config_path(), _optional_toml_config()
# These are now handled by ConfigService


def make_root_app() -> App:
    """Construct the root Cyclopts application with shared defaults.

    Returns
    -------
    App
        Root Cyclopts application configured with default parameters.
    """
    return App(
        name="codeintel",
        help="CodeIntel unified CLI for build, analytics, and serving operations.",
        default_parameter=Parameter(
            show_default=True,
        ),
        config=ConfigService.get_cyclopts_config_chain(),  # NEW: Use ConfigService
        result_action=["call_if_callable", "return_value"],
        print_error=True,
    )


# ... rest of the file remains unchanged ...
```

### 4.2 Complete Diff for cyclopts_common.py

The following lines should be **deleted** (approximately lines 41-74 in current file):

```python
# DELETE these lines:

CONFIG_ENV_PREFIX = "CODEINTEL_"
CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("codeintel.toml")

_ENV_CONFIG = cyclopts_config.Env(CONFIG_ENV_PREFIX)


def _resolve_config_path() -> Path:
    """Return the configured TOML path (env override or default).

    Returns
    -------
    Path
        Path to the config file, defaulting to ``codeintel.toml``.
    """
    env_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    return Path(env_path) if env_path else DEFAULT_CONFIG_PATH


def _optional_toml_config(apps: object, commands: tuple[str, ...], arguments: object) -> object:
    """Apply TOML config if present; otherwise return the arguments unchanged.

    Returns
    -------
    object
        Possibly updated arguments after applying TOML overrides.
    """
    path = _resolve_config_path()
    if not path.exists():
        return arguments
    toml_loader = cast("Any", cyclopts_config.Toml(str(path)))
    app_arg = cast("App", apps)
    args_arg = cast("Any", arguments)
    return toml_loader(app_arg, commands, args_arg)
```

Also **remove** these imports from the top of the file:

```python
# DELETE these imports:
import os
from typing import cast
from cyclopts import config as cyclopts_config
```

And **update** the `__all__` list to remove deleted items (if present).

---

## 5. Step 3: Remove Duplicates from cyclopts_common.py

After Step 2, verify the following are removed from `cyclopts_common.py`:

### 5.1 Removal Checklist

- [ ] `CONFIG_ENV_PREFIX` constant
- [ ] `CONFIG_PATH_ENV_VAR` constant  
- [ ] `DEFAULT_CONFIG_PATH` constant
- [ ] `_ENV_CONFIG` constant
- [ ] `_resolve_config_path()` function
- [ ] `_optional_toml_config()` function
- [ ] `import os` (if no longer needed)
- [ ] `from cyclopts import config as cyclopts_config` (if no longer needed)
- [ ] `from typing import cast` (if no longer needed)

### 5.2 Items to KEEP in cyclopts_common.py

These should remain unchanged:

- `RuntimeCLI` dataclass
- `OutputFormatCLI` dataclass
- `RuntimeCliOptions` import and re-export
- `runtime_field()` helper
- `output_field()` helper
- `runtime_cli_to_options()` function
- `build_runtime_from_cli()` function (with deprecation warning added)
- `resolve_output_format()` function
- `get_verbose()` function
- `get_output_format()` function
- `make_handler_context()` function
- All `Annotated` type aliases (`Verbose`, `ProjectRoot`, etc.)

---

## 6. Step 4: Refactor cyclopts_config.py Commands

The `cyclopts_config.py` file contains CLI commands for configuration inspection. These should be updated to use `ConfigService` where appropriate.

### 6.1 Update `cyclopts_config.py`

**Note**: The local constants in this file (`CONFIG_ENV_PREFIX`, etc.) are used for the `config show/path/env` commands and should be kept but imported from the canonical source.

```python
"""Configuration introspection commands for the CodeIntel CLI.

Provides commands to inspect effective configuration after merging
all sources (defaults, file, environment, CLI flags).
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TextIO

from cyclopts import App, Parameter

from codeintel.cli.config import (
    DEFAULT_CONFIG_PATHS,
    ConfigService,
    config_to_dict,
)
from codeintel.cli.config.service import (
    CONFIG_ENV_PREFIX,
    CONFIG_PATH_ENV_VAR,
    TOML_CONFIG_PATHS,
)

if TYPE_CHECKING:
    from codeintel.cli.cyclopts_common import RuntimeCLI


def _resolve_config_path() -> Path:
    """Return the configured TOML path (env override or default).

    Returns
    -------
    Path
        Path to the config file.
    """
    import os

    env_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    if env_path:
        return Path(env_path)

    # Check TOML paths first
    for path in TOML_CONFIG_PATHS:
        if path.exists():
            return path

    return Path("codeintel.toml")  # Default


def _load_toml_config() -> dict[str, object]:
    """Load configuration from TOML file.

    Returns
    -------
    dict[str, object]
        Configuration loaded from TOML, or empty dict if not found.
    """
    path = _resolve_config_path()
    if not path.exists():
        return {}

    with path.open("rb") as f:
        return tomllib.load(f)


def _load_env_config() -> dict[str, object]:
    """Load configuration from environment variables.

    Returns
    -------
    dict[str, object]
        Configuration values from CODEINTEL_* environment variables.
    """
    import os

    result: dict[str, object] = {}
    for key, value in os.environ.items():
        if key.startswith(CONFIG_ENV_PREFIX):
            config_key = key[len(CONFIG_ENV_PREFIX) :].lower()
            result[config_key] = value
    return result


# ... rest of file with commands remains largely unchanged ...
# The commands use local helper functions for introspection purposes


@config_app.command(name="show")
@dataclass
class ConfigShowCommand:
    """Show effective configuration after merging all sources.

    Display the merged configuration with source tracking, showing
    which values come from file, environment, or defaults.
    """

    source: Annotated[
        Literal["all", "file", "env", "cli"] | None,
        Parameter(help="Show only config from specific source."),
    ] = None
    output_format: Annotated[
        Literal["text", "json"],
        Parameter(name="--format", help="Output format."),
    ] = "text"

    def __call__(self) -> None:
        """Execute the config show command."""
        # Use ConfigService for comprehensive config loading
        service = ConfigService.load(validate=False)
        cfg_dict = config_to_dict(service.config)
        
        # Add source tracking
        sources_info = {
            "sources": list(service.sources),
        }
        
        writer = sys.stdout
        if self.output_format == "json":
            output: dict[str, object] = {"config": cfg_dict, **sources_info}
            writer.write(json.dumps(output, indent=2, default=str))
            writer.write("\n")
        else:
            writer.write("Effective Configuration:\n")
            writer.write("-" * 40 + "\n")
            for key, value in sorted(cfg_dict.items()):
                writer.write(f"{key}: {value}\n")
            writer.write("\nSources: " + " → ".join(service.sources) + "\n")


# ... other commands remain similar ...
```

### 6.2 Commands Update Summary

| Command | Changes |
|---------|---------|
| `config show` | Use `ConfigService.load()` + `config_to_dict()` |
| `config path` | Keep local `_resolve_config_path()` for TOML introspection |
| `config validate` | Can use `ConfigService.load(validate=True)` for validation |
| `config env` | Keep local `_load_env_config()` for raw env introspection |
| `config init` | Unchanged (file creation) |
| `config paths` | Update to show all paths from `ConfigService` |

---

## 7. Step 5: Add Deprecation Warnings

### 7.1 Update `build_runtime_from_cli()` in cyclopts_common.py

```python
def build_runtime_from_cli(
    options: RuntimeCliOptions | RuntimeCLI | None,
    *,
    allow_fallback: bool = True,
) -> ProjectRuntime:
    """Build a :class:`ProjectRuntime` from CLI options without Typer exits.

    .. deprecated:: 2.0
        Use ``RuntimeResolver.resolve(RuntimeParams)`` instead.
        This function will be removed in version 3.0.

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    RuntimeCliError
        If a project cannot be resolved from the provided options.
    """
    warnings.warn(
        "build_runtime_from_cli is deprecated. "
        "Use RuntimeResolver.resolve(RuntimeParams) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # ... existing implementation unchanged ...
```

### 7.2 Update `runtime_cli_to_options()` in cyclopts_common.py (Optional)

If this function is only used to support `build_runtime_from_cli()`, add a deprecation warning:

```python
def runtime_cli_to_options(
    cli: RuntimeCLI | None, *, backend: BackendFlags | None = None
) -> RuntimeCliOptions:
    """Convert a RuntimeCLI dataclass to RuntimeCliOptions.

    .. deprecated:: 2.0
        Use ``RuntimeParams.from_cyclopts()`` instead.

    Returns
    -------
    RuntimeCliOptions
        Options object suitable for runtime construction.
    """
    warnings.warn(
        "runtime_cli_to_options is deprecated. "
        "Use RuntimeParams.from_cyclopts() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # ... existing implementation ...
```

---

## 8. Step 6: Test Implementation

### 8.1 Create `tests/cli/config/test_config_integration.py`

```python
"""Integration tests for ConfigService with Cyclopts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.cli.config import ConfigService, config_to_dict
from codeintel.cli.cyclopts_common import make_root_app

if TYPE_CHECKING:
    from cyclopts import App


class TestConfigServiceCycloptsIntegration:
    """Test ConfigService integration with Cyclopts."""

    def test_get_cyclopts_config_chain_returns_list(self) -> None:
        """get_cyclopts_config_chain returns list of callables."""
        chain = ConfigService.get_cyclopts_config_chain()
        
        assert isinstance(chain, list)
        assert len(chain) == 2
        for item in chain:
            assert callable(item)

    def test_make_root_app_uses_config_service(self) -> None:
        """make_root_app creates App with ConfigService chain."""
        app = make_root_app()
        
        assert app.name == "codeintel"
        assert app.config is not None
        # Verify the config chain is from ConfigService
        assert len(app.config) == 2

    def test_config_service_respects_env_var_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigService respects CODEINTEL_CONFIG_PATH env var."""
        # Create temp config file
        config_file = tmp_path / "test-config.toml"
        config_file.write_text('[codeintel]\ncolor = false\n')
        
        monkeypatch.setenv("CODEINTEL_CONFIG_PATH", str(config_file))
        
        path = ConfigService.get_toml_config_path()
        
        assert path == config_file


class TestConfigPrecedence:
    """Test configuration precedence rules."""

    def test_env_overrides_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables override file config."""
        # Create config file with color=true
        config_file = tmp_path / "config.yaml"
        config_file.write_text("color: true\n")
        
        # Set env var to override
        monkeypatch.setenv("CODEINTEL_COLOR", "false")
        
        service = ConfigService.load(config_path=config_file, validate=False)
        
        # Env should override file
        assert service.config.color is False

    def test_cli_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI overrides have highest precedence."""
        monkeypatch.setenv("CODEINTEL_COLOR", "true")
        
        service = ConfigService.load(
            cli_overrides={"color": False},
            validate=False,
        )
        
        assert service.config.color is False

    def test_sources_tracking(self, tmp_path: Path) -> None:
        """Config sources are tracked correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("color: false\n")
        
        service = ConfigService.load(
            config_path=config_file,
            cli_overrides={"log_level": "DEBUG"},
            validate=False,
        )
        
        assert "defaults" in service.sources
        assert any("file:" in s for s in service.sources)
        assert "cli-flags" in service.sources


class TestTomlConfigPath:
    """Test TOML config path resolution."""

    def test_default_path_when_no_file(self) -> None:
        """Returns None when no config file exists."""
        # In test environment, codeintel.toml likely doesn't exist
        path = ConfigService.get_toml_config_path()
        
        # Either None or an existing path
        if path is not None:
            assert path.exists()

    def test_env_override_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CODEINTEL_CONFIG_PATH env var takes precedence."""
        config_file = tmp_path / "custom.toml"
        config_file.write_text("[codeintel]\n")
        
        monkeypatch.setenv("CODEINTEL_CONFIG_PATH", str(config_file))
        
        path = ConfigService.get_toml_config_path()
        
        assert path == config_file

    def test_search_default_locations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Searches default locations when no env override."""
        # Create codeintel.toml in a temp directory that we chdir to
        config_file = tmp_path / "codeintel.toml"
        config_file.write_text("[codeintel]\n")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            monkeypatch.delenv("CODEINTEL_CONFIG_PATH", raising=False)
            
            path = ConfigService.get_toml_config_path()
            
            assert path == config_file
        finally:
            os.chdir(original_cwd)
```

### 8.2 Create `tests/cli/config/test_deprecation_warnings.py`

```python
"""Tests for deprecation warnings in CLI functions."""

from __future__ import annotations

import warnings

import pytest

from codeintel.cli.cyclopts_common import (
    RuntimeCLI,
    build_runtime_from_cli,
    runtime_cli_to_options,
)


class TestDeprecationWarnings:
    """Test deprecation warnings are emitted."""

    def test_build_runtime_from_cli_warning(self) -> None:
        """build_runtime_from_cli emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="RuntimeResolver"):
            try:
                build_runtime_from_cli(RuntimeCLI())
            except Exception:
                # We expect this to fail since there's no project
                pass

    def test_runtime_cli_to_options_warning(self) -> None:
        """runtime_cli_to_options emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="RuntimeParams"):
            runtime_cli_to_options(RuntimeCLI())
```

### 8.3 Update `tests/cli/config/test_service.py`

Enhance existing tests:

```python
"""Tests for ConfigService."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.config import ConfigService


class TestConfigService:
    """Tests for ConfigService."""

    def test_load_defaults(self) -> None:
        """load() returns config with defaults."""
        service = ConfigService.load(validate=False)

        assert service.config is not None
        assert "defaults" in service.sources

    def test_load_with_overrides(self) -> None:
        """load() applies CLI overrides."""
        service = ConfigService.load(
            cli_overrides={"color": False},
            validate=False,
        )

        assert service.config.color is False

    def test_with_overrides(self) -> None:
        """with_overrides creates new service."""
        service = ConfigService.load(validate=False)
        original_color = service.config.color

        modified = service.with_overrides(color=not original_color)

        assert modified.config.color != original_color
        assert service.config.color == original_color  # Original unchanged

    def test_sources_tracking(self) -> None:
        """Sources are tracked correctly."""
        service = ConfigService.load(validate=False)

        # Should at least have defaults
        assert len(service.sources) >= 1

    def test_cyclopts_config_chain(self) -> None:
        """get_cyclopts_config_chain returns list."""
        chain = ConfigService.get_cyclopts_config_chain()

        assert isinstance(chain, list)
        assert len(chain) == 2  # TOML config + Env config

    def test_get_toml_config_path(self) -> None:
        """get_toml_config_path returns Path or None."""
        path = ConfigService.get_toml_config_path()
        
        # Should be None or a Path
        assert path is None or isinstance(path, Path)
```

---

## 9. Verification Checklist

After implementing all changes, verify:

### 9.1 Type Checking

```bash
uv run pyright src/codeintel/cli/config/service.py
uv run pyright src/codeintel/cli/cyclopts_common.py
uv run pyright src/codeintel/cli/cyclopts_config.py
```

### 9.2 Imports Work

```bash
# Verify all public imports work
python -c "from codeintel.cli.config import ConfigService, CONFIG_ENV_PREFIX, TOML_CONFIG_PATHS"
python -c "from codeintel.cli.cyclopts_common import make_root_app, RuntimeCLI"
```

### 9.3 App Construction

```bash
# Verify make_root_app works
python -c "from codeintel.cli.cyclopts_common import make_root_app; app = make_root_app(); print(f'App name: {app.name}')"
```

### 9.4 CLI Works

```bash
# Basic CLI smoke test
codeintel --help
codeintel config show
codeintel config path
```

### 9.5 Tests Pass

```bash
uv run pytest tests/cli/config/ -v
uv run pytest tests/cli/ -v --tb=short
```

### 9.6 No Duplicate Constants

```bash
# Should only find CONFIG_ENV_PREFIX in cli/config/
rg "CONFIG_ENV_PREFIX = " src/codeintel/cli --type py

# Expected output: only cli/config/service.py
```

### 9.7 Deprecation Warnings Work

```bash
# Test deprecation warnings in Python
python -c "
import warnings
warnings.filterwarnings('always', category=DeprecationWarning)
from codeintel.cli.cyclopts_common import build_runtime_from_cli, RuntimeCLI
try:
    build_runtime_from_cli(RuntimeCLI())
except Exception:
    pass
"
# Should see DeprecationWarning
```

---

## Summary

| Task | Files Changed | Lines Changed | Complexity |
|------|--------------|---------------|------------|
| Enhance ConfigService | `cli/config/service.py` | +60 | Low |
| Update `__init__.py` | `cli/config/__init__.py` | +5 | Low |
| Update make_root_app() | `cli/cyclopts_common.py` | ~-35 | Low |
| Refactor config commands | `cli/cyclopts_config.py` | ~+20 | Medium |
| Add deprecation warnings | `cli/cyclopts_common.py` | +15 | Low |
| Tests | `tests/cli/config/*.py` | ~+150 | Medium |
| **Total** | | **~+215 net** | **Low** |

---

## Exit Criteria for Phase 3

Phase 3 can begin when:

1. ✅ All acceptance criteria above are met
2. ✅ `make_root_app()` uses `ConfigService.get_cyclopts_config_chain()`
3. ✅ No duplicate `CONFIG_ENV_PREFIX` outside `cli/config/`
4. ✅ Deprecation warnings added to legacy functions
5. ✅ All existing CLI tests pass
6. ✅ Config precedence preserved (CLI > env > file > default)
7. ✅ Both TOML and YAML config files work

---

## Appendix: Migration Checklist

- [ ] Update `cli/config/service.py` with TOML support and Cyclopts integration
- [ ] Update `cli/config/__init__.py` with new exports
- [ ] Remove config constants from `cyclopts_common.py`
- [ ] Remove config functions from `cyclopts_common.py`
- [ ] Update `make_root_app()` to use `ConfigService`
- [ ] Remove unused imports from `cyclopts_common.py`
- [ ] Refactor `cyclopts_config.py` commands to use `ConfigService`
- [ ] Add deprecation warning to `build_runtime_from_cli()`
- [ ] Add deprecation warning to `runtime_cli_to_options()`
- [ ] Create `tests/cli/config/test_config_integration.py`
- [ ] Create `tests/cli/config/test_deprecation_warnings.py`
- [ ] Update `tests/cli/config/test_service.py`
- [ ] Run type checking
- [ ] Run full CLI test suite
- [ ] Manual smoke test: `codeintel --help`
- [ ] Manual smoke test: `codeintel config show`
- [ ] Manual smoke test: `codeintel build status` (with project)
- [ ] Verify env var override works: `CODEINTEL_COLOR=false codeintel config show`
