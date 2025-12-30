def func_a(x: int, y: int) -> int:
    from pkg.mod_b import func_b
    return func_b(x) + y
