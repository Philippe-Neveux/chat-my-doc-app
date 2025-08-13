"""
RAG (Retrieval Augmented Generation) Service

This module provides a comprehensive RAG service that combines query processing,
document retrieval from Qdrant, and context formatting for LLM consumption.
It handles the complete pipeline from user query to formatted context.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from chat_my_doc_app.config import get_config
from chat_my_doc_app.db import QdrantService


class DocumentSource:
    """Represents a source document with metadata for citations."""

    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any], score: float):
        self.id = doc_id
        self.content = content
        self.metadata = metadata
        self.score = score
        self.citation_id: Optional[int] = None  # Will be set when generating citations

    def __repr__(self):
        return f"DocumentSource(id={self.id}, score={self.score:.3f})"


class RAGService:
    """Service for Retrieval Augmented Generation pipeline."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG service.

        Args:
            config_dict: Configuration dictionary. If None, loads from default config.
        """
        if config_dict is None:
            config_dict = get_config()

        self.config = config_dict
        self.rag_config = config_dict.get('rag', {})

        # Initialize Qdrant service
        self.qdrant_service = QdrantService(config_dict)

        # RAG configuration
        self.max_context_length = self.rag_config.get('max_context_length')
        self.context_overlap = self.rag_config.get('context_overlap')

        # Retrieval settings
        retrieval_config = self.rag_config.get('retrieval', {})
        self.chunk_size = retrieval_config.get('chunk_size')
        self.chunk_overlap = retrieval_config.get('chunk_overlap')
        self.min_chunk_size = retrieval_config.get('min_chunk_size')

        # Generation settings
        generation_config = self.rag_config.get('generation', {})
        self.include_sources = generation_config.get('include_sources')
        self.source_format = generation_config.get('source_format', 'markdown')

        logger.info("RAGService initialized successfully")

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess and clean user query for better retrieval.

        Args:
            query: Raw user query

        Returns:
            Cleaned and optimized query
        """
        if not query or not query.strip():
            return ""

        return query.strip()

    def retrieve_documents(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[DocumentSource]:
        """
        Retrieve relevant documents from Qdrant.

        Args:
            query: Search query
            limit: Maximum number of documents to retrieve
            score_threshold: Minimum relevance score
            metadata_filter: Optional metadata filtering

        Returns:
            List of DocumentSource objects
        """
        try:
            # Preprocess query
            processed_query = self.preprocess_query(query)
            if not processed_query:
                logger.warning("Empty query after preprocessing")
                return []

            # Perform search
            search_results = self.qdrant_service.similarity_search(
                query=processed_query,
                limit=limit,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter
            )

            # Convert to DocumentSource objects
            documents = []
            for result_data, score in search_results:
                payload = result_data.get('payload', {})

                # Extract content from IMDB review payload
                content = payload.get('review') or str(payload)

                # Create document source
                doc_source = DocumentSource(
                    doc_id=str(result_data.get('id', 'unknown')),
                    content=content,
                    metadata=payload,
                    score=score
                )
                documents.append(doc_source)

            logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def format_documents_for_context(self, documents: List[DocumentSource]) -> str:
        """
        Format retrieved documents into context string for LLM.

        Args:
            documents: List of DocumentSource objects

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            # Assign citation ID
            doc.citation_id = i

            # Extract IMDB review metadata for rich formatting
            metadata = doc.metadata
            movie_title = metadata.get('title', 'Unknown Movie')
            review_title = metadata.get('review_title', '')
            review_rating = metadata.get('review_rating', 'N/A')
            movie_rating = metadata.get('rating', 'N/A')
            genre = metadata.get('genre', '')
            year = metadata.get('year', '')

            # Create header with movie information
            movie_info_parts = []
            if movie_title != 'Unknown Movie':
                movie_info_parts.append(f"Movie: {movie_title}")
            if year:
                movie_info_parts.append(f"Year: {year}")
            if genre:
                movie_info_parts.append(f"Genre: {genre}")
            if movie_rating != 'N/A':
                movie_info_parts.append(f"Rating: {movie_rating}")

            movie_info = " | ".join(movie_info_parts) if movie_info_parts else ""

            # Format document content with rich metadata
            if self.source_format == 'markdown':
                header = f"**Source {i}** (Score: {doc.score:.3f})"
                if movie_info:
                    header += f"\n*{movie_info}*"
                if review_title:
                    header += f"\n**Review Title:** {review_title}"
                if review_rating != 'N/A':
                    header += f" (Rating: {review_rating}/10)"
                doc_text = f"{header}\n\n{doc.content}\n"
            else:
                header = f"Source {i} (Score: {doc.score:.3f})"
                if movie_info:
                    header += f"\n{movie_info}"
                if review_title:
                    header += f"\nReview Title: {review_title}"
                if review_rating != 'N/A':
                    header += f" (Rating: {review_rating}/10)"
                doc_text = f"{header}\n\n{doc.content}\n"

            # Check if adding this document would exceed context limit
            if current_length + len(doc_text) > self.max_context_length:
                # Try to fit a truncated version
                remaining_space = self.max_context_length - current_length - 200  # Reserve space for truncation notice
                if remaining_space > self.min_chunk_size:
                    truncated_content = doc.content[:remaining_space] + "..."
                    if self.source_format == 'markdown':
                        truncated_doc_text = f"{header}\n\n{truncated_content}\n*(Truncated)*\n"
                    else:
                        truncated_doc_text = f"{header}\n\n{truncated_content}\n(Truncated)\n"
                    context_parts.append(truncated_doc_text)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        # Join all parts
        context = "\n".join(context_parts)

        logger.debug(f"Formatted context length: {len(context)} characters")
        return context

    def generate_citations(self, documents: List[DocumentSource]) -> List[Dict[str, Any]]:
        """
        Generate citation information for retrieved documents.

        Args:
            documents: List of DocumentSource objects with citation_id set

        Returns:
            List of citation dictionaries
        """
        citations = []

        for doc in documents:
            if doc.citation_id is None:
                continue

            citation = {
                'id': doc.citation_id,
                'document_id': doc.id,
                'score': doc.score,
                'metadata': doc.metadata.copy()
            }

            # Add IMDB review metadata
            metadata = doc.metadata
            if 'title' in metadata:
                citation['movie_title'] = metadata['title']
            if 'review_title' in metadata:
                citation['review_title'] = metadata['review_title']
            if 'review_rating' in metadata:
                citation['review_rating'] = metadata['review_rating']
            if 'rating' in metadata:
                citation['movie_rating'] = metadata['rating']
            if 'genre' in metadata:
                citation['genre'] = metadata['genre']
            if 'year' in metadata:
                citation['year'] = metadata['year']

            citations.append(citation)

        return citations

    def retrieve_context(
        self,
        query: str,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_citations: Optional[bool] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Complete RAG retrieval pipeline: query → search → format → citations.

        Args:
            query: User query
            limit: Max documents to retrieve (uses config default if None)
            score_threshold: Min relevance score (uses config default if None)
            metadata_filter: Optional metadata filtering
            include_citations: Whether to include citations (uses config default if None)

        Returns:
            Tuple of (formatted_context, citations_list)
        """
        try:
            # Use defaults from config if not specified
            if limit is None:
                search_config = self.config.get('qdrant', {}).get('search', {})
                limit = search_config.get('default_limit', 5)

            if score_threshold is None:
                search_config = self.config.get('qdrant', {}).get('search', {})
                score_threshold = search_config.get('default_score_threshold', 0.0)

            if include_citations is None:
                include_citations = self.include_sources

            documents = self.retrieve_documents(
                query=query,
                limit=limit,  # type: ignore
                score_threshold=score_threshold,  # type: ignore
                metadata_filter=metadata_filter
            )

            if not documents:
                return "No relevant information found for your query.", []

            # Format context
            context = self.format_documents_for_context(documents)

            # Generate citations if requested
            citations = []
            if include_citations:
                citations = self.generate_citations(documents)

            logger.info(f"RAG retrieval completed: {len(documents)} docs, {len(context)} chars context")

            return context, citations

        except Exception as e:
            logger.error(f"Error in RAG retrieval pipeline: {str(e)}")
            error_context = f"Error retrieving information for your query: {str(e)}"
            return error_context, []

    def search_by_movie_genre(
        self,
        genre: str,
        limit: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Search for reviews by movie genre.

        Args:
            genre: Movie genre to search for
            limit: Maximum number of results

        Returns:
            Tuple of (formatted_context, citations_list)
        """
        metadata_filter = {'genre': genre}
        return self.retrieve_context(
            query=f"{genre} movies",
            limit=limit,
            metadata_filter=metadata_filter
        )

    def search_by_movie_year(
        self,
        year: str,
        query: str = "movie review",
        limit: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Search for reviews by movie year.

        Args:
            year: Movie year to search for
            query: Optional query text
            limit: Maximum number of results

        Returns:
            Tuple of (formatted_context, citations_list)
        """
        metadata_filter = {'year': str(year)}
        return self.retrieve_context(
            query=query,
            limit=limit,
            metadata_filter=metadata_filter
        )

    def search_by_rating_range(
        self,
        min_rating: float,
        max_rating: float = 10.0,
        query: str = "movie review",
        limit: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Search for highly-rated movies (this would require range filtering support in Qdrant).
        For now, this is a placeholder that does a regular search.

        Args:
            min_rating: Minimum review rating
            max_rating: Maximum review rating
            query: Query text
            limit: Maximum number of results

        Returns:
            Tuple of (formatted_context, citations_list)
        """
        # Note: Qdrant range filtering would need to be implemented separately
        # For now, we'll do a regular search and filter in post-processing
        context, citations = self.retrieve_context(query, limit=limit * 2)  # Get more results for filtering

        # Filter citations by rating range
        filtered_citations = []
        for citation in citations:
            review_rating = citation.get('review_rating')
            if review_rating and isinstance(review_rating, (int, float)):
                if min_rating <= review_rating <= max_rating:
                    filtered_citations.append(citation)

        # Take only the requested number
        filtered_citations = filtered_citations[:limit]

        return context, filtered_citations
