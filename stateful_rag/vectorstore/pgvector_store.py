import psycopg2
import psycopg2.extras
import numpy as np
from stateful_rag.config import get_settings
from stateful_rag.ingestion.chunker import DocumentChunk


def get_connection():
    """Raw psycopg2 connection — used for all DB operations."""
    settings = get_settings()
    return psycopg2.connect(settings.postgres_url)


def add_chunks(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
    """
    Store chunks + their embeddings in pgvector.
    Skips duplicates based on chunk_id (idempotent).

    Args:
        chunks: Output from chunk_document()
        embeddings: One embedding vector per chunk, same order

    Returns:
        Number of new chunks inserted
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have same length")

    inserted = 0

    conn = get_connection()
    try:
        cur = conn.cursor()

        for chunk, embedding in zip(chunks, embeddings):
            cur.execute("""
                INSERT INTO documents (chunk_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
            """, (
                chunk.chunk_id,
                chunk.text,
                psycopg2.extras.Json(chunk.metadata),
                embedding,
            ))

            if cur.rowcount > 0:
                inserted += 1

        conn.commit()
        print(f"  Inserted {inserted} new chunks "
              f"({len(chunks) - inserted} already existed)")

    finally:
        conn.close()

    return inserted


def similarity_search(
    query_embedding: list[float],
    k: int = 4,
    filter_source: str | None = None,
) -> list[dict]:
    """
    Find top-k most similar chunks using cosine distance.

    Args:
        query_embedding: Embedded query vector
        k: Number of results to return
        filter_source: Optional filename to restrict search to one document

    Returns:
        List of dicts with content, metadata, and similarity score
    """
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if filter_source:
            cur.execute("""
                SELECT
                    chunk_id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE metadata->>'source' = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, filter_source, query_embedding, k))
        else:
            cur.execute("""
                SELECT
                    chunk_id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, k))

        results = cur.fetchall()
        return [dict(row) for row in results]

    finally:
        conn.close()


def get_collection_stats() -> dict:
    """Summary of what's stored in pgvector."""
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM documents")
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT metadata->>'source' AS source, COUNT(*) AS chunks
            FROM documents
            GROUP BY metadata->>'source'
            ORDER BY chunks DESC
        """)
        per_source = cur.fetchall()

        return {
            "total_chunks": total,
            "sources": [
                {"source": row[0], "chunks": row[1]}
                for row in per_source
            ]
        }
    finally:
        conn.close()


def delete_source(source_name: str) -> int:
    """Remove all chunks from a specific document."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM documents
            WHERE metadata->>'source' = %s
        """, (source_name,))
        deleted = cur.rowcount
        conn.commit()
        print(f"  Deleted {deleted} chunks for '{source_name}'")
        return deleted
    finally:
        conn.close()


# <=> operator in pgvector
# ---------------------------
# <=> is the cosine distance operator in pgvector. 1 - (embedding <=> query) converts distance to similarity — higher is better. Other operators:

# <-> — L2 (Euclidean) distance (what ChromaDB used by default)
# <#> — negative inner product
# <=> — cosine distance ← we use this, best for text embeddings

# ON CONFLICT (chunk_id) DO NOTHING
# ------------------------------------
# This is PostgreSQL's upsert syntax. If a row with the same chunk_id already exists, skip it silently. Same idempotent ingestion behavior as Level 1 but handled at the database level — more reliable than checking in Python first.