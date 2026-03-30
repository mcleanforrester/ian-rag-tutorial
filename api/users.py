from identity.models import User, PUBLIC, INTERNAL, CONFIDENTIAL

USERS = {
    "alice": User(name="Alice", department="engineering", permission_level=INTERNAL),
    "bob": User(name="Bob", department="engineering", permission_level=CONFIDENTIAL),
    "carol": User(name="Carol", department="accounting", permission_level=INTERNAL),
    "dave": User(name="Dave", department="hr", permission_level=CONFIDENTIAL),
    "eve": User(name="Eve", department="accounting", permission_level=PUBLIC),
}
