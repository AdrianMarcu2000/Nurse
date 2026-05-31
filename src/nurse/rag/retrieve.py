"""Query the LanceDB index at runtime to retrieve relevant protocol chunks."""
from __future__ import annotations

import logging
from functools import lru_cache

from nurse.config import get_config, resolve

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_db_and_embedder():
    import lancedb
    from sentence_transformers import SentenceTransformer

    cfg = get_config()["rag"]
    db = lancedb.connect(str(resolve(cfg["db_path"])))
    embedder = SentenceTransformer(cfg["embedding_model"])
    return db, embedder


def retrieve(query: str) -> str:
    """
    Retrieve the top-k relevant protocol chunks for a query.
    Returns a formatted string ready for injection into the prompt,
    or an empty string if the index is unavailable or no results pass threshold.
    """
    cfg = get_config()["rag"]
    table_name = cfg["table_name"]
    top_k = cfg["top_k"]
    threshold = cfg["score_threshold"]

    try:
        db, embedder = _get_db_and_embedder()
        if table_name not in db.table_names():
            return ""
        table = db.open_table(table_name)
        query_vec = embedder.encode(query).tolist()
        results = (
            table.search(query_vec)
            .limit(top_k)
            .to_list()
        )
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        return ""

    # LanceDB returns _distance; lower = more similar (cosine)
    passages = [
        r["text"] for r in results
        if r.get("_distance", 1.0) < (1.0 - threshold)
    ]

    if not passages:
        return ""

    return "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
