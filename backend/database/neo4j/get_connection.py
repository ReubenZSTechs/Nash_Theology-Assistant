import os
from functools import lru_cache

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()


@lru_cache(maxsize=1)
def get_neo4j_driver() -> Driver:
    """
    Return one shared Neo4j driver for the whole process.

    A Neo4j driver keeps its own connection pool and is safe to share between
    threads, so it is created once and reused (same env vars and defaults as
    backend/services/push_bible_neo4j_service.py).
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "nash-password"))

    # By default the driver keeps retrying for 30 s when Neo4j is unreachable.
    # Retrieval is interactive, so give up quickly and let the pipeline report the error.
    return GraphDatabase.driver(uri, auth=auth, connection_timeout=5.0, max_transaction_retry_time=3.0)
