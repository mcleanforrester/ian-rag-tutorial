from langchain_chroma import Chroma
from langchain_core.documents import Document

from identity.models import User, LEVELS


class DocumentRepository:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

    def find_relevant(self, query: str, user: User, k: int = 4) -> list[Document]:
        """Find documents relevant to the query, filtered by user's department and permissions."""
        allowed_levels = [
            level.name
            for level in LEVELS.values()
            if user.permission_level.can_access(level)
        ]

        return self.vector_store.similarity_search(
            query,
            k=k,
            filter={
                "$and": [
                    {"department": user.department},
                    {"permission_level": {"$in": allowed_levels}},
                ]
            },
        )
