def parse_multi_value(cell) -> list[str]:
    """Parse a CSV cell into a list of string values.

    Accepts comma-separated strings ("a, b, c") and scalar values like int/float.
    """
    if cell is None:
        return []

    # Pandas can pass NaN/nullable values here.
    try:
        if cell != cell:
            return []
    except Exception:
        pass

    if isinstance(cell, str):
        return [x.strip() for x in cell.split(",") if x.strip()]

    return [str(cell).strip()]
