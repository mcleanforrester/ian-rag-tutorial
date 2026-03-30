import re

from identity.models import LEVELS, PermissionLevel


def parse_permission_level(filename: str) -> PermissionLevel:
    """Extract permission level from a PDF filename.

    The filename must contain one of: public, internal, confidential.
    Raises ValueError if no recognized level is found.
    """
    name_lower = filename.lower()
    match = re.search(r"\b(public|internal|confidential)\b", name_lower)
    if match:
        return LEVELS[match.group(1)]
    raise ValueError(
        f"Cannot determine permission level from filename '{filename}'. "
        f"Filename must contain one of: {', '.join(LEVELS.keys())}"
    )
