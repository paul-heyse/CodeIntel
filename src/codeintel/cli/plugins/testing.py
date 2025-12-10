"""Plugin testing utilities.

Provide a test harness for plugin developers to test
their plugins in isolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from codeintel.cli.plugins.manifest import PluginCapability, PluginManifest
from codeintel.cli.plugins.sandbox import PluginSandbox, SandboxConfig


@runtime_checkable
class OperationSpecProtocol(Protocol):
    """Protocol for operation specs.

    Defines the expected interface for operation specs
    registered by plugins.
    """

    @property
    def operation_id(self) -> str:
        """Get operation ID."""
        ...


@dataclass
class PluginTestResult:
    """Result of a plugin test.

    Parameters
    ----------
    success
        Whether test passed.
    message
        Result message.
    errors
        List of errors.
    warnings
        List of warnings.
    """

    success: bool
    message: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PluginTestHarness:
    """Test harness for plugin development.

    Provide isolated environment for testing plugins
    without affecting the global CLI state.

    Parameters
    ----------
    manifest
        Plugin manifest.
    capabilities
        Override capabilities for testing.
    """

    def __init__(
        self,
        manifest: PluginManifest,
        capabilities: set[PluginCapability] | None = None,
    ) -> None:
        """Initialize test harness."""
        self._manifest = manifest
        self._capabilities = capabilities or set(manifest.capabilities)
        self._registered_operations: list[str] = []
        self._test_results: list[PluginTestResult] = []

    @property
    def manifest(self) -> PluginManifest:
        """Get plugin manifest.

        Returns
        -------
        PluginManifest
            The plugin manifest.
        """
        return self._manifest

    @property
    def registered_operations(self) -> list[str]:
        """Get list of registered operations.

        Returns
        -------
        list[str]
            Operation IDs registered by the plugin.
        """
        return list(self._registered_operations)

    def validate_manifest(self) -> PluginTestResult:
        """Validate plugin manifest.

        Returns
        -------
        PluginTestResult
            Validation result.
        """
        errors = self._manifest.validate()

        if errors:
            return PluginTestResult(
                success=False,
                message="Manifest validation failed",
                errors=errors,
            )

        return PluginTestResult(
            success=True,
            message="Manifest is valid",
        )

    def test_load(self) -> PluginTestResult:
        """Test plugin loading.

        Returns
        -------
        PluginTestResult
            Load test result.
        """
        config = SandboxConfig(allowed_capabilities=self._capabilities)

        try:
            with PluginSandbox(self._manifest, config) as sandbox:
                module = sandbox.load_plugin()

                # Check for required attributes
                warnings: list[str] = []
                if not hasattr(module, "register"):
                    warnings.append("Plugin has no 'register' function")

                return PluginTestResult(
                    success=True,
                    message=f"Plugin loaded successfully: {module.__name__}",
                    warnings=warnings,
                )

        except ImportError as e:
            return PluginTestResult(
                success=False,
                message="Plugin failed to load",
                errors=[str(e)],
            )
        except (RuntimeError, AttributeError, TypeError) as e:
            return PluginTestResult(
                success=False,
                message="Plugin loading raised exception",
                errors=[f"{type(e).__name__}: {e}"],
            )

    def test_operations(self) -> PluginTestResult:
        """Test plugin operations.

        Returns
        -------
        PluginTestResult
            Operations test result.
        """
        config = SandboxConfig(allowed_capabilities=self._capabilities)
        errors: list[str] = []
        warnings: list[str] = []

        try:
            with PluginSandbox(self._manifest, config) as sandbox:
                module = sandbox.load_plugin()

                if not hasattr(module, "register"):
                    return PluginTestResult(
                        success=True,
                        message="No operations to test",
                        warnings=["Plugin has no 'register' function"],
                    )

                # Create mock registry that captures registered operations
                registered: list[object] = []

                def mock_register(spec: object) -> object:
                    """Register an operation spec.

                    Parameters
                    ----------
                    spec
                        Operation spec to register.

                    Returns
                    -------
                    object
                        The registered spec.
                    """
                    registered.append(spec)
                    return spec

                class MockRegistry:
                    """Mock registry for testing."""

                    register = staticmethod(mock_register)

                # Register operations
                module.register(MockRegistry())

                # Validate registered operations
                for spec in registered:
                    if not isinstance(spec, OperationSpecProtocol):
                        errors.append("Operation missing operation_id")
                        continue

                    op_id = spec.operation_id
                    if not op_id.startswith(f"{self._manifest.name}."):
                        warnings.append(
                            f"Operation '{op_id}' should be prefixed "
                            f"with plugin name '{self._manifest.name}.'",
                        )

                    self._registered_operations.append(op_id)

                return PluginTestResult(
                    success=len(errors) == 0,
                    message=f"Registered {len(registered)} operations",
                    errors=errors,
                    warnings=warnings,
                )

        except (ImportError, RuntimeError, AttributeError, TypeError) as e:
            return PluginTestResult(
                success=False,
                message="Operation testing failed",
                errors=[f"{type(e).__name__}: {e}"],
            )

    def run_all_tests(self) -> list[PluginTestResult]:
        """Run all plugin tests.

        Returns
        -------
        list[PluginTestResult]
            All test results.
        """
        results = [
            self.validate_manifest(),
            self.test_load(),
            self.test_operations(),
        ]
        self._test_results = results
        return results

    def get_summary(self) -> dict[str, Any]:
        """Get test summary.

        Returns
        -------
        dict[str, Any]
            Test summary.
        """
        passed = sum(1 for r in self._test_results if r.success)
        failed = len(self._test_results) - passed

        return {
            "plugin": self._manifest.name,
            "version": self._manifest.version,
            "tests_run": len(self._test_results),
            "passed": passed,
            "failed": failed,
            "registered_operations": self._registered_operations,
        }


def create_plugin_scaffold(
    name: str,
    output_dir: Path,
    *,
    capabilities: list[PluginCapability] | None = None,
) -> Path:
    """Create plugin scaffold.

    Parameters
    ----------
    name
        Plugin name.
    output_dir
        Output directory.
    capabilities
        Initial capabilities.

    Returns
    -------
    Path
        Path to created plugin directory.
    """
    capabilities = capabilities or [PluginCapability.REGISTER_OPERATIONS]

    plugin_dir = output_dir / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "name": name,
        "version": "0.1.0",
        "api_version": "1.0.0",
        "description": f"{name.title()} plugin for CodeIntel CLI",
        "author": "",
        "capabilities": [cap.value for cap in capabilities],
        "dependencies": [],
        "entry_point": f"{name}.main",
    }

    manifest_path = plugin_dir / "plugin.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Create package
    pkg_dir = plugin_dir / name
    pkg_dir.mkdir(exist_ok=True)

    # Create __init__.py
    init_content = f'''"""
{name.title()} plugin for CodeIntel CLI.
"""

from {name}.main import register

__all__ = ["register"]
'''
    (pkg_dir / "__init__.py").write_text(init_content, encoding="utf-8")

    # Create main.py - uses current CLI architecture patterns
    main_content = f'''"""
Main module for {name} plugin.

This module follows the CLI's HandlerContext + OperationSpec pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.execution import OperationSpec

if TYPE_CHECKING:
    from codeintel.cli.handlers.context import HandlerContext


def _example_handler(ctx: HandlerContext) -> CliResult[dict[str, str]]:
    """Example operation handler.

    Parameters
    ----------
    ctx
        Handler context with params and resources.

    Returns
    -------
    CliResult[dict[str, str]]
        Result with greeting message.
    """
    # Access params via ctx.param_str(), ctx.param_int(), etc.
    _ = ctx  # Acknowledge context (remove when adding real logic)
    return CliResult.ok({{"message": "Hello from {name}!"}})


def register(registry: object) -> None:
    """Register plugin operations with the CLI.

    Parameters
    ----------
    registry
        Operation registry (provides register method).

    Notes
    -----
    Operations must follow the OperationSpec pattern with:
    - operation_id: Unique ID prefixed with plugin name
    - name: Human-readable display name
    - description: Help text
    - handler: Function accepting HandlerContext
    - group: Command group for organization
    """
    registry.register(
        OperationSpec(
            operation_id="{name}.hello",
            name="Hello",
            description="Example operation from {name} plugin",
            handler=_example_handler,
            group="{name}",
        )
    )
'''
    (pkg_dir / "main.py").write_text(main_content, encoding="utf-8")

    # Create tests directory
    tests_dir = plugin_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create test file
    test_content = f'''"""
Tests for {name} plugin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.plugins import PluginManifest, PluginTestHarness


@pytest.fixture
def manifest() -> PluginManifest:
    """Load plugin manifest."""
    manifest_path = Path(__file__).parent.parent / "plugin.json"
    return PluginManifest.load(manifest_path)


def test_manifest_valid(manifest: PluginManifest) -> None:
    """Test manifest is valid."""
    harness = PluginTestHarness(manifest)
    result = harness.validate_manifest()
    assert result.success, result.errors


def test_plugin_loads(manifest: PluginManifest) -> None:
    """Test plugin loads successfully."""
    harness = PluginTestHarness(manifest)
    result = harness.test_load()
    assert result.success, result.errors


def test_operations_register(manifest: PluginManifest) -> None:
    """Test operations register correctly."""
    harness = PluginTestHarness(manifest)
    result = harness.test_operations()
    assert result.success, result.errors
'''
    (tests_dir / f"test_{name}.py").write_text(test_content, encoding="utf-8")

    # Create README
    readme_content = f"""# {name.title()} Plugin

A plugin for CodeIntel CLI.

## Installation

Copy this plugin to your plugins directory:

```bash
cp -r {name} ~/.codeintel/plugins/
```

## Usage

```bash
codeintel op call {name}.hello
```

## Development

Run tests:

```bash
pytest tests/
```
"""
    (plugin_dir / "README.md").write_text(readme_content, encoding="utf-8")

    return plugin_dir


__all__ = [
    "OperationSpecProtocol",
    "PluginTestHarness",
    "PluginTestResult",
    "create_plugin_scaffold",
]
