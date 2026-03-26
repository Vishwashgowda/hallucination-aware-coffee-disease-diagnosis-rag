"""
Retrieval Evaluator Module (CRAG)
Filters and scores retrieved chunks for relevance
"""

from typing import List, Dict, Tuple
import re


class RetrievalEvaluator:
    def __init__(self, relevance_threshold: float = 0.3):
        """
        Initialize Retrieval Evaluator

        Args:
            relevance_threshold: Threshold for filtering irrelevant documents (0-1)
        """
        self.relevance_threshold = relevance_threshold

    def evaluate_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """
        Evaluate and filter chunks for relevance

        Args:
            chunks: List of retrieved document chunks
            query: Original query

        Returns:
            Filtered and scored chunks, sorted by relevance
        """
        scored_chunks = []

        for chunk in chunks:
            score = self._calculate_relevance_score(chunk, query)

            scored_chunks.append({
                'content': chunk['content'],
                'source': chunk['source'],
                'metadata': chunk.get('metadata', {}),
                'relevance_score': score
            })

        # Filter by threshold
        filtered = [c for c in scored_chunks if c['relevance_score'] >= self.relevance_threshold]

        # Sort by score (descending)
        filtered.sort(key=lambda x: x['relevance_score'], reverse=True)

        return filtered

    def _calculate_relevance_score(self, chunk: Dict, query: str) -> float:
        """
        Calculate relevance score for a chunk

        Args:
            chunk: Document chunk
            query: Query string

        Returns:
            Relevance score (0-1)
        """
        content = chunk['content'].lower()
        query_lower = query.lower()

        score = 0.0

        # Extract query keywords (remove common words)
        query_keywords = self._extract_keywords(query_lower)

        if not query_keywords:
            return 0.1  # Minimum score for non-keyword queries

        # Count keyword matches
        matches = 0
        for keyword in query_keywords:
            if keyword in content:
                matches += 1

        # Calculate match ratio
        keyword_score = matches / len(query_keywords) if query_keywords else 0

        # Check for disease-specific terms
        disease_terms = [
            'coffee', 'disease', 'leaf', 'rust', 'borer', 'scale', 'mite',
            'rot', 'wilt', 'blight', 'anthracnose', 'spot', 'symptom',
            'treatment', 'management', 'control', 'prevention'
        ]

        disease_match = sum(1 for term in disease_terms if term in content)
        disease_score = min(disease_match / 3, 1.0)  # Normalize

        # Combined score
        score = (keyword_score * 0.6) + (disease_score * 0.4)

        return min(score, 1.0)

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'do', 'does',
            'what', 'how', 'why', 'when', 'where', 'which', 'my', 'your', 'our'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', query)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        return keywords

    def filter_by_source(self, chunks: List[Dict], source_filter: List[str] = None) -> List[Dict]:
        """
        Filter chunks by source file

        Args:
            chunks: List of chunks
            source_filter: List of allowed source files

        Returns:
            Filtered chunks
        """
        if not source_filter:
            return chunks

        return [c for c in chunks if c['source'] in source_filter]

    def get_top_chunks(self, chunks: List[Dict], k: int = 3) -> List[Dict]:
        """
        Get top-k most relevant chunks

        Args:
            chunks: List of scored chunks
            k: Number of chunks to return

        Returns:
            Top-k chunks
        """
        return sorted(
            chunks,
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )[:k]

    def is_relevant(self, chunk: Dict, query: str) -> bool:
        """Check if a chunk is relevant to the query"""
        score = self._calculate_relevance_score(chunk, query)
        return score >= self.relevance_threshold
