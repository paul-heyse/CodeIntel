"""Codemod that converts tuple-based analytics rows to TypedDict constructors."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import libcst as cst
import libcst.matchers as m
from libcst import CodemodContext
from libcst.codemod import CodemodCommand, parallel_exec_transform_with_pretty_print

from tools.analytics_refactors.tuple_row_config import ALL_SPECS, TupleRowSpec


@dataclass(frozen=True)
class ImportTarget:
    """Resolved import target for a fully qualified symbol."""

    QUALNAME_MIN_PARTS: ClassVar[int] = 2
    module: str
    name: str

    @classmethod
    def parse(cls, qualname: str) -> ImportTarget:
        """
        Split a fully qualified name into module and attribute.

        Returns
        -------
        ImportTarget
            Parsed import target.

        Raises
        ------
        ValueError
            If the qualified name does not contain a module and attribute.
        """
        parts = qualname.rsplit(".", maxsplit=1)
        if len(parts) < cls.QUALNAME_MIN_PARTS:
            message = f"Invalid qualname: {qualname}"
            raise ValueError(message)
        return cls(module=parts[0], name=parts[1])


class _TupleRowBodyTransformer(cst.CSTTransformer):
    """Rewrite rows.append((...)) to TypedDict construction inside target builders."""

    def __init__(self, spec: TupleRowSpec) -> None:
        self.spec = spec
        self.in_target_function: list[bool] = []
        self.row_constructor_used = False
        self.row_import_present = False

    def on_visit(self, node: cst.CSTNode) -> bool:
        """
        Track function scope and imports during traversal.

        Returns
        -------
        bool
            True to continue traversal.
        """
        if isinstance(node, cst.ImportFrom):
            target = ImportTarget.parse(self.spec.row_type_qualname)
            if m.matches(
                node,
                m.ImportFrom(
                    module=m.Attribute() | m.Name(),
                    names=m.OneOrMore(
                        m.ImportAlias(
                            name=m.Name(target.name) | m.Name(self.spec.row_type_local),
                        )
                    ),
                ),
            ):
                self.row_import_present = True
        if isinstance(node, cst.FunctionDef):
            self.in_target_function.append(node.name.value in self.spec.builder_functions)
        return True

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:
        """
        Rewrite tuple appends when exiting nodes.

        Returns
        -------
        cst.CSTNode
            Possibly transformed node.
        """
        node_out: cst.CSTNode = updated_node
        if isinstance(original_node, cst.FunctionDef):
            self.in_target_function.pop()
            return node_out

        if (
            isinstance(updated_node, cst.Call)
            and any(self.in_target_function)
            and self._is_target_append(updated_node)
        ):
            tuple_arg = updated_node.args[0].value
            elements = [element.value for element in tuple_arg.elements]
            if len(elements) == len(self.spec.field_names):
                row_call = cst.Call(
                    func=cst.Name(self.spec.row_type_local),
                    args=[
                        cst.Arg(keyword=cst.Name(field_name), value=value)
                        for field_name, value in zip(
                            self.spec.field_names, elements, strict=True
                        )
                    ],
                )
                self.row_constructor_used = True
                node_out = updated_node.with_changes(args=[cst.Arg(value=row_call)])
        return node_out

    def _is_target_append(self, call: cst.Call) -> bool:
        """
        Return True when the call matches rows.append((...)) for this spec.

        Returns
        -------
        bool
            True when the call is an append to the target rows list.
        """
        func = call.func
        return (
            isinstance(func, cst.Attribute)
            and isinstance(func.value, cst.Name)
            and func.value.value == self.spec.rows_var
            and func.attr.value == "append"
            and len(call.args) == 1
            and isinstance(call.args[0].value, cst.Tuple)
        )


class TupleRowToDictTransform(CodemodCommand):
    """Convert tuple-based row appends to TypedDict construction for a module."""

    DESCRIPTION: str = "Convert tuple row builders to TypedDict constructors"

    def __init__(self, context: CodemodContext, spec: TupleRowSpec) -> None:
        super().__init__(context)
        self.spec = spec

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        """
        Apply the tuple-to-dict transform and inject imports when needed.

        Returns
        -------
        cst.Module
            Transformed module.
        """
        transformer = _TupleRowBodyTransformer(self.spec)
        new_tree = tree.visit(transformer)

        if transformer.row_constructor_used and not transformer.row_import_present:
            target = ImportTarget.parse(self.spec.row_type_qualname)
            import_stmt = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name.from_value(target.module),
                        names=[
                            cst.ImportAlias(
                                name=cst.Name(target.name),
                                asname=cst.AsName(cst.Name(self.spec.row_type_local)),
                            )
                        ],
                    )
                ]
            )
            new_tree = new_tree.with_changes(body=[import_stmt, *new_tree.body])

        return new_tree


def _spec_for_module(module_name: str) -> TupleRowSpec | None:
    """
    Return the spec matching a module name when present.

    Returns
    -------
    TupleRowSpec | None
        Matching spec or None.
    """
    for spec in ALL_SPECS:
        if spec.module == module_name:
            return spec
    return None


class MultiSpecDriver(CodemodCommand):
    """Dispatch codemod execution based on configured specs."""

    DESCRIPTION: str = "Apply tuple→dict refactors for configured analytics datasets."

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        """
        Delegate to the inner transform when the module matches a spec.

        Returns
        -------
        cst.Module
            Transformed module or original when no spec matches.
        """
        filename = self.context.filename
        if filename is None:
            return tree
        module_name = filename.replace("/", ".").removesuffix(".py")
        spec = _spec_for_module(module_name)
        if spec is None:
            return tree
        inner = TupleRowToDictTransform(self.context, spec)
        return inner.transform_module_impl(tree)


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point to run the codemod over the provided paths."""
    parser = argparse.ArgumentParser(
        description="Refactor tuple-based analytics rows to TypedDict constructors.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to run the codemod on.",
    )
    args = parser.parse_args(argv)
    parallel_exec_transform_with_pretty_print(MultiSpecDriver, args.paths, None)


if __name__ == "__main__":
    main()
