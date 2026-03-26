"""
Vector Store Module
Manages FAISS index for efficient similarity search
"""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class FAISSVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_db_path: str = "vector_db"):
        """
        Initialize FAISS Vector Store

        Args:
            model_name: Sentence-transformer model to use
            vector_db_path: Path to store/load FAISS index
        """
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dim = 384 if "L6" in model_name else 768  # embedding dimension

    def create_index(self, documents: List) -> None:
        """
        Create FAISS index from documents

        Args:
            documents: List of langchain documents with page_content
        """
        print(f"Creating embeddings for {len(documents)} documents...")

        # Extract text from documents
        texts = [doc.page_content for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)

        self.documents = documents
        print(f"Index created with {len(documents)} documents")

        # Save index
        self._save_index()

    def _save_index(self) -> None:
        """Save FAISS index to disk"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        index_path = os.path.join(self.vector_db_path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}")

    def _load_index(self) -> bool:
        """Load FAISS index from disk"""
        index_path = os.path.join(self.vector_db_path, "faiss_index.bin")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Index loaded from {index_path}")
            return True
        return False

    def search(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar documents

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not initialized. Create index first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:  # Valid index
                results.append((self.documents[idx], float(distance)))

        return results

    def retrieve_top_k(self, query: str, k: int = 5) -> List:
        """Retrieve top-k documents for query"""
        results = self.search(query, k)
        return [doc for doc, _ in results]
