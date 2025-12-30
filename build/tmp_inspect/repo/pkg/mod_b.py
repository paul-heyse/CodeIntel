def func_b(x: int) -> int:
    from pkg.mod_c import func_c
    total = x * 2
    for value in func_c():
        total += value
    return total
