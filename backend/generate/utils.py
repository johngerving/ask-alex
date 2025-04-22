def validate_brackets(s: str) -> bool:
    """Validate if the string has balanced brackets."""
    depth = 0
    for c in s:
        depth += c == "["
        depth -= c == "]"
        if depth < 0:
            return False

    return depth == 0
