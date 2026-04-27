"""
Vector memory store backed by ChromaDB.

Stores analysis findings and retrieves semantically similar context
to inject into new analysis sessions.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """
    Persistent semantic memory using ChromaDB + sentence-transformers.
    Falls back to a simple in-memory list if ChromaDB is unavailable.
    """

    def __init__(self, persist_dir: Path | str | None = None) -> None:
        self._persist_dir = str(persist_dir) if persist_dir else None
        self._collection = None
        self._fallback: list[dict[str, Any]] = []
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            if self._persist_dir:
                client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                client = chromadb.Client()

            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = client.get_or_create_collection(
                name="analyst_memory",
                embedding_function=ef,
            )
            logger.info("chroma_memory_initialised", persist_dir=self._persist_dir)
        except ImportError:
            logger.warning("chromadb_not_installed_using_fallback_memory")
        except Exception as exc:
            logger.warning("chroma_init_failed", error=str(exc))

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a text chunk and return its ID."""
        doc_id = str(uuid.uuid4())[:12]
        if self._collection is not None:
            self._collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata or {}],
            )
        else:
            self._fallback.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        logger.debug("memory_stored", id=doc_id, text_preview=text[:60])
        return doc_id

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Retrieve top-k semantically similar memories."""
        if self._collection is not None:
            results = self._collection.query(query_texts=[query], n_results=k)
            docs = results.get("documents", [[]])[0]
            return docs
        # Fallback: keyword overlap
        query_terms = set(query.lower().split())
        scored = [
            (len(query_terms & set(m["text"].lower().split())), m["text"])
            for m in self._fallback
        ]
        scored.sort(key=lambda x: -x[0])
        return [text for _, text in scored[:k] if _ > 0]

    def clear(self) -> None:
        if self._collection is not None:
            self._collection.delete(where={"id": {"$ne": ""}})
        self._fallback.clear()
        logger.info("memory_cleared")
