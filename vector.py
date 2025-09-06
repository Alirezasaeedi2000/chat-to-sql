import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


LOGGER = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


@dataclass
class RetrievedContext:
    texts: List[str]
    metadatas: List[Dict[str, Any]]


class QueryExpander:
    """Lightweight synonym expansion to improve retrieval without changing SQL generation semantics."""

    def __init__(self, synonyms_path: str) -> None:
        self.synonyms_path = synonyms_path
        self.synonyms: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self.synonyms_path):
            try:
                with open(self.synonyms_path, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            except Exception as exc:
                LOGGER.warning("Failed to load synonyms file %s: %s", self.synonyms_path, exc)
                self.synonyms = {}

    def expand(self, query_text: str) -> str:
        """Append synonym hints to the query string to enrich retrieval context."""
        if not self.synonyms:
            return query_text
        hints: List[str] = []
        lowered = query_text.lower()
        for canonical, syns in self.synonyms.items():
            if canonical.lower() in lowered:
                hints.extend(syns)
        if hints:
            return f"{query_text}\n\nAlso consider related terms: {', '.join(sorted(set(hints)))}"
        return query_text


class VectorStoreManager:
    """Manages embedding, persistence, and retrieval for schema, samples, and past queries."""

    def __init__(
        self,
        persist_directory: str = ".vector_store",
        collection_name: str = "mysql_nl2sql",
        embedding_model: str = "mxbai-embed-large",
        synonyms_path: str = os.path.join("outputs", "synonyms.json"),
    ) -> None:
        ensure_dir(persist_directory)
        ensure_dir("outputs")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.query_expander = QueryExpander(synonyms_path)

    @property
    def vector_store(self) -> Chroma:
        return self._vector_store

    def _schema_snippet(self, engine: Engine, table_name: str) -> str:
        inspector = inspect(engine)
        cols = inspector.get_columns(table_name)
        pk = inspector.get_pk_constraint(table_name)
        pk_cols = pk.get("constrained_columns") or []
        col_desc = ", ".join([f"{c['name']}({str(c.get('type'))})" for c in cols])
        snippet = (
            f"Table `{table_name}` columns: {col_desc}. "
            f"Primary key: {', '.join(pk_cols) if pk_cols else 'None'}."
        )
        fks = inspector.get_foreign_keys(table_name)
        if fks:
            fk_desc = "; ".join(
                [
                    f"{','.join(fk.get('constrained_columns', []))} -> {fk.get('referred_table')}({','.join(fk.get('referred_columns', []))})"
                    for fk in fks
                ]
            )
            snippet += f" Foreign keys: {fk_desc}."
        return snippet

    def _sample_rows_snippet(self, engine: Engine, table_name: str, sample_rows: int) -> Tuple[str, Optional[pd.DataFrame]]:
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(f"SELECT * FROM `{table_name}` LIMIT :lim"), conn, params={"lim": sample_rows})
            if df.empty:
                return f"No sample rows available for `{table_name}`.", df
            head = df.head(min(sample_rows, 5))
            sample_text = head.to_csv(index=False)
            return f"Sample rows for `{table_name}` (CSV):\n{sample_text}", df
        except Exception as exc:
            LOGGER.warning("Failed to sample rows for %s: %s", table_name, exc)
            return f"Failed to sample rows for `{table_name}` due to error.", None

    def upsert_schema_and_samples(self, engine: Engine, sample_rows_per_table: int = 5, include_views: bool = True) -> int:
        """Indexes schema metadata and sample rows into the vector store. Returns number of items added."""
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if include_views:
            try:
                tables.extend([v for v in inspector.get_view_names() if v not in tables])
            except Exception:
                pass

        added = 0
        for table_name in tables:
            try:
                # Delete previous entries for the table to avoid duplication.
                try:
                    self._vector_store.delete(where={"table": table_name})
                except Exception:
                    pass

                schema_text = self._schema_snippet(engine, table_name)
                self._vector_store.add_texts(
                    texts=[schema_text],
                    metadatas=[{"type": "schema", "table": table_name}],
                )
                added += 1

                sample_text, _ = self._sample_rows_snippet(engine, table_name, sample_rows_per_table)
                self._vector_store.add_texts(
                    texts=[sample_text],
                    metadatas=[{"type": "sample", "table": table_name}],
                )
                added += 1
            except Exception as exc:
                LOGGER.error("Failed to index table %s: %s", table_name, exc)
        try:
            # Ensure persistence on disk
            self._vector_store.persist()
        except Exception:
            pass
        return added

    def add_past_query(self, sql: str, result_summary: Optional[str] = None, note: Optional[str] = None) -> None:
        text_block = f"SQL: {sql}"
        if result_summary:
            text_block += f"\nResult: {result_summary}"
        if note:
            text_block += f"\nNote: {note}"
        try:
            self._vector_store.add_texts(
                texts=[text_block],
                metadatas=[{"type": "past_query"}],
            )
            self._vector_store.persist()
        except Exception as exc:
            LOGGER.warning("Failed to add past query to vector store: %s", exc)

    def similarity_search(self, query_text: str, top_k: int = 8) -> RetrievedContext:
        enriched = self.query_expander.expand(query_text)
        try:
            docs = self._vector_store.similarity_search(enriched, k=top_k)
        except Exception as exc:
            LOGGER.warning("Vector search failed, returning empty context: %s", exc)
            return RetrievedContext(texts=[], metadatas=[])
        texts = [d.page_content for d in docs]
        metas = [d.metadata or {} for d in docs]
        return RetrievedContext(texts=texts, metadatas=metas)

