"""
Node 1 — understand the query.

Before searching, we prepare two things from the raw question:
  1. an embedding vector, used by both the vector search and the graph seed search;
  2. any Bible references written in the question (e.g. "Romans 3:28", "Roma 3:23-24"),
     which become guaranteed seeds for the graph search.
"""
from backend.core.decorators.log_node_call import log_node
from backend.pipelines.rag.states import RAGState
from backend.services.embedding_service import get_embedding_service
from backend.utils.sermon_postgres_configs import BOOK_LOOKUP, MAX_MENTION_SPAN, MENTION


def extract_verse_ids(text: str) -> list[str]:
    """
    Find Bible references in free text and turn them into verse ids ("Book.Chapter.Verse").

    Uses the same regular expression as the sermon ingestion step, so it understands both
    English and Indonesian book names. Ranges ("John 3:16-18") are expanded verse by verse.
    Ids that do not exist (e.g. "John 3:99") are filtered out later by Neo4j.
    """
    verse_ids = []
    for match in MENTION.finditer(text or ""):
        book_name = " ".join(match.group("book").lower().split())
        book = BOOK_LOOKUP[book_name]
        chapter = int(match.group("chapter"))
        start = int(match.group("start"))
        end = int(match.group("end") or start)

        # Ignore broken or very long ranges and keep only the first verse.
        if end < start or end - start >= MAX_MENTION_SPAN:
            end = start

        for verse in range(start, end + 1):
            verse_id = f"{book}.{chapter}.{verse}"
            if verse_id not in verse_ids:
                verse_ids.append(verse_id)

    return verse_ids


@log_node("rag.understand_query")
def understand_query(state: RAGState) -> dict:
    query = state["query"]
    return {
        "query_vector": get_embedding_service().embed_query(query),
        "explicit_verse_ids": extract_verse_ids(query),
    }
