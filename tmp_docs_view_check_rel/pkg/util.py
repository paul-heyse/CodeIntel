from pkg.mod import hello


def loud(name: str) -> str:
    msg = hello(name)
    return msg.upper()
