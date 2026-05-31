"""Build the LanceDB RAG index from markdown/text files in data/protocols/."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from nurse.config import get_config, resolve

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 400   # characters
_CHUNK_OVERLAP = 80


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker on sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > size and current:
            chunks.append(current.strip())
            # Keep last `overlap` chars for context continuity
            current = current[-overlap:] + " " + sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_index(force: bool = False) -> None:
    """Index all .md and .txt files in data/protocols/ into LanceDB."""
    import lancedb
    from sentence_transformers import SentenceTransformer

    cfg = get_config()["rag"]
    db_path = resolve(cfg["db_path"])
    table_name = cfg["table_name"]
    protocols_dir = resolve("data/protocols")

    db = lancedb.connect(str(db_path))

    if table_name in db.table_names() and not force:
        logger.info("RAG index already exists. Use force=True to rebuild.")
        return

    docs = list(protocols_dir.glob("*.md")) + list(protocols_dir.glob("*.txt"))
    if not docs:
        logger.warning("No protocol documents found in %s", protocols_dir)
        return

    logger.info("Loading embedding model %s …", cfg["embedding_model"])
    embedder = SentenceTransformer(cfg["embedding_model"])

    records = []
    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_text(text)):
            embedding = embedder.encode(chunk).tolist()
            records.append({
                "text": chunk,
                "source": doc_path.name,
                "chunk_id": i,
                "vector": embedding,
            })

    if table_name in db.table_names():
        db.drop_table(table_name)

    db.create_table(table_name, data=records)
    logger.info("Indexed %d chunks from %d documents.", len(records), len(docs))
