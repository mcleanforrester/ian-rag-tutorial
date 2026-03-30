from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionLevel:
    name: str
    rank: int

    def can_access(self, other: "PermissionLevel") -> bool:
        """Can a user at this level access a document at the other level?"""
        return other.rank <= self.rank


PUBLIC = PermissionLevel("public", 0)
INTERNAL = PermissionLevel("internal", 1)
CONFIDENTIAL = PermissionLevel("confidential", 2)

LEVELS = {"public": PUBLIC, "internal": INTERNAL, "confidential": CONFIDENTIAL}


@dataclass
class User:
    name: str
    department: str
    permission_level: PermissionLevel
