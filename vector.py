# vector.py
# Handles the creation and management of the vector store for RAG.
# It embeds schema info, sample rows, and past queries for context retrieval.

import os
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Import the MCP handler to safely access the database
import mcp_handler

# --- Configuration ---
# Use a file path for the FAISS index and its metadata
VECTOR_STORE_PATH = "vector_store"
INDEX_FILE = os.path.join(VECTOR_STORE_PATH, "vectors.index")
METADATA_FILE = os.path.join(VECTOR_STORE_PATH, "metadata.json")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - VECTOR - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self, embedding_model_name='mxbai-embed-large'):
        """
        Initializes the VectorStore, loading the embedding model and the store from disk if available.
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.metadata = []  # List of dicts with info about each vector

        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.load_from_disk()
        else:
            logging.info("No existing vector store found. A new one will be created.")
            # Initialize an empty index. The dimension comes from the model.
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_dim)

    def save_to_disk(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, INDEX_FILE)
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f)
            logging.info(f"Vector store saved to {VECTOR_STORE_PATH}")

    def load_from_disk(self):
        """Loads the FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logging.info(f"Vector store loaded successfully from {VECTOR_STORE_PATH}")
        except Exception as e:
            logging.error(f"Failed to load vector store from disk: {e}. Re-initializing.")
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata = []

    def add_documents(self, documents):
        """
        Embeds and adds a list of documents to the vector store.
        Each document is a dictionary with 'content' and 'type' keys.
        """
        if not documents:
            return

        contents = [doc['content'] for doc in documents]
        embeddings = self.model.encode(contents, convert_to_tensor=False, show_progress_bar=True)

        # Ensure embeddings are in the correct format (float32) for FAISS
        embeddings = np.array(embeddings, dtype='float32')

        self.index.add(embeddings)
        self.metadata.extend(documents)
        logging.info(f"Added {len(documents)} new documents to the vector store.")

    def retrieve_context(self, query, k=5):
        """
        Retrieves the top-k most relevant documents for a given query.
        """
        if self.index.ntotal == 0:
            logging.warning("Vector store is empty. Cannot retrieve context.")
            return []

        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype='float32')

        distances, indices = self.index.search(query_embedding, k)

        # Filter out invalid indices (if k > number of docs)
        valid_indices = [i for i in indices[0] if i != -1]

        # Get unique results
        seen_content = set()
        results = []
        for idx in valid_indices:
            doc = self.metadata[idx]
            if doc['content'] not in seen_content:
                results.append(doc)
                seen_content.add(doc['content'])

        return results

def prime_vector_store(vector_store_instance):
    """

    Primes the vector store with schema information and sample rows from the database.
    This is a critical step for providing context to the LLM.
    """
    logging.info("Priming vector store with database schema and sample data...")

    # Use the MCP handler to safely get schema
    schema_response = mcp_handler.get_schema()
    if schema_response.get("status") != "success":
        logging.error("Failed to retrieve schema. Vector store priming aborted.")
        return

    schema_data = schema_response.get("data", {})
    documents = []

    for table, columns in schema_data.items():
        # 1. Add schema information
        col_defs = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
        schema_content = f"Table '{table}' has columns: {col_defs}."
        documents.append({"type": "schema", "table": table, "content": schema_content})

        # 2. Add sample rows
        sample_rows_response = mcp_handler.get_sample_rows(table)
        if sample_rows_response.get("status") == "success":
            rows = sample_rows_response.get("data", [])
            if rows:
                # Convert rows to a string format for embedding
                sample_content = f"Sample rows from table '{table}': {json.dumps(rows, indent=2)}"
                documents.append({"type": "sample_rows", "table": table, "content": sample_content})

    if documents:
        vector_store_instance.add_documents(documents)
        vector_store_instance.save_to_disk() # Save after priming
    else:
        logging.warning("No documents were generated during priming. Is the database empty?")

def add_query_to_history(vector_store_instance, user_query, sql_query):
    """
    Adds a successful user query and its corresponding SQL to the vector store.
    This helps the RAG system learn from interactions.
    """
    logging.info("Adding successful query to vector store history.")
    content = f"User question: '{user_query}' was answered with SQL: `{sql_query}`"
    document = {"type": "past_query", "content": content}
    vector_store_instance.add_documents([document])
    vector_store_instance.save_to_disk() # Save after adding history
