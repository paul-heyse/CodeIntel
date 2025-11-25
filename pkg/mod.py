"""Helpers for sample fixture functions."""


def hello(name: str) -> str:
    """Generate a greeting message.

    Parameters
    ----------
    name:
        Person to greet.

    Returns
    -------
    str
        Friendly greeting.
    """
    return f"hi {name}"


def adder(x: int, y: int) -> int:
    """Add two integers.

    Parameters
    ----------
    x:
        First operand.
    y:
        Second operand.

    Returns
    -------
    int
        Sum of the two operands.
    """
    return x + y
