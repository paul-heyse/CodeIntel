"""Base recipe executor for composable workflows.

This module provides the abstract base class for recipe executors across
all domains (analytics, graphs, ingestion). The base class defines common
patterns for scratch space management and cleanup.

Architecture Note
-----------------
Domain-specific recipe executors do not formally extend BaseRecipeExecutor
because each domain has distinct execution patterns:

- Analytics: Plan-based execution (plan() + execute()), sequential only
- Graphs: Stage-based execution with parallel support
- Ingestion: Stage-based execution with parallel support, different context

However, all domain executors follow these common patterns:
- Scratch space management via PluginScratch
- Plugin context building
- Result aggregation
- Optional parallel execution via ThreadPoolExecutor (graphs, ingestion)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from codeintel.core.plugins.context import PluginScratch


class BaseRecipeExecutor[R, C, Result](ABC):
    """Abstract base class for recipe executors.

    Provide common infrastructure for recipe execution including scratch
    space management and cleanup. Domain-specific executors implement
    the execute method with their specific execution logic.

    Type Parameters
    ---------------
    R
        Recipe type (e.g., Recipe, IngestRecipe).
    C
        Context type (e.g., RecipeExecutionContext, RecipeExecutorContext).
    Result
        Result type (e.g., RecipeExecutionReport, RecipeExecutionResult).

    Attributes
    ----------
    _scratch
        Shared scratch space for inter-plugin communication.

    Extension Points
    ----------------
    Domain-specific recipe executors should:

    1. Override `__init__` to accept domain-specific dependencies
    2. Implement `execute()` with domain-specific execution logic
    3. Call `_cleanup()` in a finally block to clean up scratch space

    Notes
    -----
    While domain executors do not formally extend this base (due to
    differing execution patterns), they should follow the documented
    patterns for consistency.

    Examples
    --------
    >>> class MyRecipeExecutor(BaseRecipeExecutor[MyRecipe, MyContext, MyResult]):
    ...     def execute(self, recipe: MyRecipe, context: MyContext) -> MyResult:
    ...         try:
    ...             # Execute plugins using self._scratch
    ...             return MyResult(success=True)
    ...         finally:
    ...             self._cleanup()
    """

    def __init__(self, scratch: PluginScratch | None = None) -> None:
        """Initialize the executor with optional scratch space.

        Parameters
        ----------
        scratch
            Shared scratch space for inter-plugin communication.
            If not provided, a new scratch space is created.
        """
        self._scratch = scratch or PluginScratch()

    @abstractmethod
    def execute(self, recipe: R, context: C) -> Result:
        """Execute a recipe and return results.

        Subclasses must implement this method with their domain-specific
        execution logic. The implementation should:

        1. Iterate through recipe plugins or stages
        2. Build plugin contexts using domain-specific builders
        3. Execute plugins with appropriate error handling
        4. Aggregate results into the result type
        5. Call `_cleanup()` in a finally block

        Parameters
        ----------
        recipe
            Recipe defining plugins to execute.
        context
            Execution context with dependencies and configuration.

        Returns
        -------
        Result
            Execution result with plugin records and status.
        """
        ...

    def _cleanup(self) -> None:
        """Clean up scratch space.

        Call this method in a finally block after recipe execution
        to ensure scratch space resources are released.
        """
        self._scratch.cleanup()


__all__ = [
    "BaseRecipeExecutor",
]
