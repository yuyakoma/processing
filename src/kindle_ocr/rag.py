"""RAG ingestion and retrieval helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class RAGStore:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.persist_directory = Path(cfg.rag.vector_store)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=cfg.book_id,
            metadata={"book_id": cfg.book_id},
        )
        self.embedder = SentenceTransformer(cfg.rag.embedding_model)

    def ingest(self, chunks: Iterable[tuple[str, dict[str, str]]]) -> None:
        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for index, (text, metadata) in enumerate(chunks, start=1):
            if not text.strip():
                continue
            documents.append(text)
            metadatas.append(metadata)
            ids.append(f"{self.cfg.book_id}-{index}")
            embeddings.append(self.embedder.encode(text, convert_to_numpy=True).tolist())

        if not documents:
            logger.warning("No documents to ingest for %s", self.cfg.book_id)
            return

        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
        logger.info("Ingested %d text chunks for %s", len(documents), self.cfg.book_id)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )

        payload = []
        for docs, metas, dists in zip(
            results.get("documents", []),
            results.get("metadatas", []),
            results.get("distances", []),
        ):
            for doc, meta, dist in zip(docs, metas, dists):
                payload.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": float(dist),
                })
        return payload
