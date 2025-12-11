"""Fluent builder for FunctionContext test data.

This module provides a builder class for constructing FunctionContext
instances with sensible defaults and a fluent interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from codeintel.analytics.compute.semantic_roles.classification import FunctionContext

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class FunctionContextBuilder:
    """Fluent builder for FunctionContext test data.

    Create FunctionContext instances with sensible defaults for testing
    semantic role classification logic.

    Attributes
    ----------
    goid : int
        Function global object ID.
    rel_path : str
        Relative path to the file containing the function.
    qualname : str
        Fully qualified function name.
    decorators : list[str]
        List of decorator names applied to the function.
    effects : dict[str, object]
        Side effects detected in the function.
    contracts : dict[str, object]
        Contract metadata (preconditions, postconditions).
    module_tags : list[str]
        Tags from the parent module.
    module_name : str | None
        Name of the parent module.
    graph : dict[str, int]
        Graph metrics (fan_in, fan_out, etc.).
    loc : int | None
        Lines of code in the function.

    Examples
    --------
    >>> builder = FunctionContextBuilder()
    >>> ctx = builder.with_decorators("@app.route").with_module_tags("api").build()
    >>> ctx.decorators
    ['@app.route']
    """

    goid: int = 0
    rel_path: str = "pkg/api.py"
    qualname: str = "pkg.api.fn"
    decorators: list[str] = field(default_factory=list)
    effects: dict[str, object] = field(default_factory=dict)
    contracts: dict[str, object] = field(default_factory=dict)
    module_tags: list[str] = field(default_factory=list)
    module_name: str | None = "pkg.api"
    graph: dict[str, int] = field(default_factory=dict)
    loc: int | None = 10

    def with_goid(self, goid: int) -> Self:
        """Set the function global object ID.

        Parameters
        ----------
        goid
            Function global object ID.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.goid = goid
        return self

    def with_rel_path(self, rel_path: str) -> Self:
        """Set the relative path to the file.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.rel_path = rel_path
        return self

    def with_qualname(self, qualname: str) -> Self:
        """Set the fully qualified function name.

        Parameters
        ----------
        qualname
            Fully qualified name (e.g., "pkg.module.func").

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.qualname = qualname
        return self

    def with_decorators(self, *decorators: str) -> Self:
        """Set the function decorators.

        Parameters
        ----------
        *decorators
            Decorator names to apply.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.decorators = list(decorators)
        return self

    def with_module_tags(self, *tags: str) -> Self:
        """Set module tags.

        Parameters
        ----------
        *tags
            Tag names to apply.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.module_tags = list(tags)
        return self

    def with_module_name(self, name: str | None) -> Self:
        """Set the module name.

        Parameters
        ----------
        name
            Module name or None.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.module_name = name
        return self

    def with_effects(self, effects: Mapping[str, object]) -> Self:
        """Set detected side effects.

        Parameters
        ----------
        effects
            Effect mapping.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.effects = dict(effects)
        return self

    def with_contracts(self, contracts: Mapping[str, object]) -> Self:
        """Set contract metadata.

        Parameters
        ----------
        contracts
            Contract mapping.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.contracts = dict(contracts)
        return self

    def with_graph(self, graph: Mapping[str, int]) -> Self:
        """Set graph metrics.

        Parameters
        ----------
        graph
            Graph metrics mapping (e.g., {"fan_in": 5, "fan_out": 3}).

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.graph = dict(graph)
        return self

    def with_loc(self, loc: int | None) -> Self:
        """Set lines of code.

        Parameters
        ----------
        loc
            Lines of code or None.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self.loc = loc
        return self

    def build(self) -> FunctionContext:
        """Build the FunctionContext instance.

        Returns
        -------
        FunctionContext
            Constructed FunctionContext with configured values.
        """
        return FunctionContext(
            goid=self.goid,
            rel_path=self.rel_path,
            qualname=self.qualname,
            decorators=self.decorators,
            effects=self.effects,
            contracts=self.contracts,
            module_tags=self.module_tags,
            module_name=self.module_name,
            graph=self.graph,
            loc=self.loc,
        )


__all__ = ["FunctionContextBuilder"]
