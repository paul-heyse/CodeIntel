"""Deliberately small module used to exercise tooling diagnostics."""


def bad_type(x: int) -> int:
    """
    Return a value with an incorrect type to trigger static errors.

    Returns
    -------
    int
        The incorrectly typed result.
    """
    return x + "a"


def add(x: int, y: int) -> int:
    """
    Add two integers.

    Returns
    -------
    int
        The summed value.
    """
    return x + y
