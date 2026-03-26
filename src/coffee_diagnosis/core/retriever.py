"""
Retriever Module
Retrieves relevant context from vector store based on query and conversation history
"""

from typing import List, Dict, Optional
from .vector_store import FAISSVectorStore


class Retriever:
    def __init__(self, vector_store: FAISSVectorStore, top_k: int = 5):
        """
        Initialize Retriever

        Args:
            vector_store: FAISSVectorStore instance
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, previous_answers: List[str] = None) -> List[Dict]:
        """
        Retrieve relevant documents based on query and history

        Args:
            query: Current user query
            previous_answers: List of previous user answers (conversation history)

        Returns:
            List of retrieved documents with metadata
        """
        # Combine query with context from previous answers
        if previous_answers:
            context_str = " ".join(previous_answers)
            combined_query = f"{query} {context_str}"
        else:
            combined_query = query

        # Retrieve documents
        docs = self.vector_store.retrieve_top_k(combined_query, k=self.top_k)

        # Format results
        results = []
        for doc in docs:
            results.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source_file', 'unknown'),
                'metadata': doc.metadata
            })

        return results

    def retrieve_with_scores(self, query: str, previous_answers: List[str] = None) -> List[Dict]:
        """
        Retrieve documents with relevance scores

        Args:
            query: Current user query
            previous_answers: List of previous user answers

        Returns:
            List of retrieved documents with relevance scores
        """
        if previous_answers:
            context_str = " ".join(previous_answers)
            combined_query = f"{query} {context_str}"
        else:
            combined_query = query

        # Search with scores
        search_results = self.vector_store.search(combined_query, k=self.top_k)

        results = []
        for doc, score in search_results:
            results.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source_file', 'unknown'),
                'metadata': doc.metadata,
                'score': score  # Lower score = more similar
            })

        return results

    def format_context(self, documents: List[Dict], max_length: int = 2000) -> str:
        """
        Format retrieved documents into context string

        Args:
            documents: List of retrieved documents
            max_length: Maximum total length

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for doc in documents:
            content = doc['content']
            source = doc['source']

            text = f"Source: {source}\n{content}\n"

            if current_length + len(text) > max_length:
                break

            context_parts.append(text)
            current_length += len(text)

        return "\n---\n".join(context_parts)
