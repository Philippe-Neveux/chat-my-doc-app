"""
RAG (Retrieval Augmented Generation) Service

This module provides a comprehensive RAG service that combines query processing,
document retrieval from Qdrant, and context formatting for LLM consumption.
It handles the complete pipeline from user query to formatted context.
"""

import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from loguru import logger

from chat_my_doc_app.config import get_config
from chat_my_doc_app.db import QdrantService
from chat_my_doc_app.llms import GeminiChat


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


class RetrievalService:
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

        logger.info("RetrievalService initialized successfully")

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


class RAGImdbState(TypedDict):
    """State structure for the RAG workflow."""
    query: str
    context: str
    citations: List[Dict[str, Any]]
    response: str
    metadata: Dict[str, Any]


class RAGImdb:
    """LangGraph workflow for RAG processing."""

    def __init__(
        self,
        model_name: str,
        config_dict: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize RAG workflow.

        Args:
            model_name: Name of the Gemini model to use (e.g., "gemini-2.0-flash-lite")
            config_dict: Configuration dictionary. If None, loads from default config.
            system_prompt: Custom system prompt. If None, uses default movie review assistant prompt.
        """
        if config_dict is None:
            config_dict = get_config()

        self.config = config_dict
        self.rag_config = config_dict.get('rag', {})
        self.model_name = model_name

        # Initialize RAG service
        self.rag_service = RetrievalService(config_dict)

        # Initialize LLM (GeminiChat will get API URL from environment)
        self.llm = GeminiChat(
            model_name=model_name,
            system_prompt=system_prompt or self.get_default_system_prompt()
        )

        # Build the workflow graph
        self.workflow = self._build_workflow()

        logger.info("RAGImdb initialized successfully")

    @staticmethod
    def get_default_system_prompt() -> str:
        """Get default system prompt for the LLM."""
        return """You are a helpful movie review assistant. You help users find and understand movie reviews from the IMDB dataset.

Instructions:
1. Use the provided context to answer user questions about movies and reviews
2. Be specific and cite information from the reviews when possible
3. If the context doesn't contain relevant information, say so clearly
4. Focus on movie-related queries: reviews, ratings, genres, years, actors, etc.
5. Provide helpful and informative responses based on the review data

Always be honest about what information is available in the context."""

    def _build_workflow(self):
        """Build the LangGraph workflow."""

        # Define the workflow graph
        workflow = (
            StateGraph(RAGImdbState)
            # Define nodes for each step in the workflow
            .add_node("retrieve", self.retrieve_node)
            .add_node("generate", self.generate_node)
            .add_node("respond", self.respond_node)
            # Define edges
            .set_entry_point("retrieve")
            .add_edge("retrieve", "generate")
            .add_edge("generate", "respond")
            .add_edge("respond", END)
        ).compile()

        return workflow

    def retrieve_node(self, state: RAGImdbState) -> RAGImdbState:
        """
        Retrieve relevant documents using RAG service.

        Args:
            state: Current workflow state

        Returns:
            Updated state with context and citations
        """
        try:
            logger.info(f"Retrieving context for query: '{state['query'][:50]}...'")

            # Get retrieval parameters from config
            generation_config = self.rag_config.get('generation', {})
            search_config = self.config.get('qdrant', {}).get('search', {})

            limit = search_config.get('default_limit', 5)
            score_threshold = search_config.get('default_score_threshold', 0.0)
            include_citations = generation_config.get('include_sources', True)

            # Use RAG service to retrieve context
            context, citations = self.rag_service.retrieve_context(
                query=state['query'],
                limit=limit,
                score_threshold=score_threshold,
                include_citations=include_citations
            )

            # Update state
            return {
                **state,
                "context": context,
                "citations": citations,
                "metadata": {
                    **state.get("metadata", {}),
                    "retrieved_docs": len(citations),
                    "context_length": len(context)
                }
            }

        except Exception as e:
            logger.error(f"Error in retrieve node: {str(e)}")
            return {
                **state,
                "context": f"Error retrieving information: {str(e)}",
                "citations": [],
                "metadata": {
                    **state.get("metadata", {}),
                    "retrieve_error": str(e)
                }
            }

    def generate_node(self, state: RAGImdbState) -> RAGImdbState:
        """
        Generate response using LLM with retrieved context.

        Args:
            state: Current workflow state

        Returns:
            Updated state with generated response
        """
        try:
            logger.info("Generating response using LLM")

            # Create prompt with context
            prompt = self._create_generation_prompt(state['query'], state['context'])

            # Generate response using LLM
            messages = cast(List[BaseMessage], [HumanMessage(content=prompt)])
            result = self.llm._generate(messages)

            # Extract generated text
            content = result.generations[0].message.content
            if isinstance(content, str):
                generated_response = content
            else:
                generated_response = str(content)

            return {
                **state,
                "response": generated_response,
                "metadata": {
                    **state.get("metadata", {}),
                    "generated": True,
                    "response_length": len(generated_response)
                }
            }

        except Exception as e:
            logger.error(f"Error in generate node: {str(e)}")
            return {
                **state,
                "response": f"Error generating response: {str(e)}",
                "metadata": {
                    **state.get("metadata", {}),
                    "generate_error": str(e)
                }
            }

    def respond_node(self, state: RAGImdbState) -> RAGImdbState:
        """
        Format final response with optional citations.

        Args:
            state: Current workflow state

        Returns:
            Final state with formatted response
        """
        try:
            logger.info("Formatting final response")

            response = state['response']
            citations = state.get('citations', [])

            # Add citations if enabled and available
            generation_config = self.rag_config.get('generation', {})
            if generation_config.get('include_sources', True) and citations:
                response = self._add_citations_to_response(response, citations)

            return {
                **state,
                "response": response,
                "metadata": {
                    **state.get("metadata", {}),
                    "final_response_length": len(response),
                    "citations_included": len(citations) > 0
                }
            }

        except Exception as e:
            logger.error(f"Error in respond node: {str(e)}")
            return {
                **state,
                "response": state.get('response', f"Error formatting response: {str(e)}"),
                "metadata": {
                    **state.get("metadata", {}),
                    "respond_error": str(e)
                }
            }

    def _create_generation_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for LLM generation.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the following context from movie reviews, please answer the user's question.

Context:
{context}

User Question: {query}

Please provide a helpful and informative answer based on the context above. If the context doesn't contain enough information to fully answer the question, please say so and provide what information is available."""

        return prompt

    def _add_citations_to_response(self, response: str, citations: List[Dict[str, Any]]) -> str:
        """
        Add citation information to the response.

        Args:
            response: Generated response
            citations: List of citation dictionaries

        Returns:
            Response with citations appended
        """
        if not citations:
            return response

        # Add citations section
        citation_text = "\n\n**Sources:**\n"

        for citation in citations:
            citation_id = citation.get('id', 'Unknown')
            score = citation.get('score', 0.0)

            # Add movie information if available
            movie_title = citation.get('movie_title', 'Unknown Movie')
            review_title = citation.get('review_title', '')
            year = citation.get('year', '')
            genre = citation.get('genre', '')

            citation_line = f"{citation_id}. **{movie_title}**"

            if year:
                citation_line += f" ({year})"
            if genre:
                citation_line += f" - {genre}"
            if review_title:
                citation_line += f"\n   Review: \"{review_title}\""

            citation_line += f" (Relevance: {score:.2f})"
            citation_text += citation_line + "\n"

        return response + citation_text

    def process_query(self, query: str) -> RAGImdbState:
        """
        Process a user query through the complete RAG workflow.

        Args:
            query: User query string

        Returns:
            Dictionary containing response and metadata
        """
        logger.info(f"Processing query through RAG workflow: '{query[:50]}...'")

        # Initialize workflow state
        initial_state = {
            "query": query,
            "context": "",
            "citations": [],
            "response": "",
            "metadata": {
                "query_length": len(query),
                "workflow_started": True
            }
        }

        try:
            # Run the workflow
            final_state = self.workflow.invoke(cast(RAGImdbState, initial_state))

            logger.info("RAG workflow completed successfully")

            return {
                "query": str(initial_state["query"]),
                "response": str(final_state["response"]),
                "citations": final_state.get("citations", []),
                "context": str(final_state.get("context", "")),
                "metadata": final_state.get("metadata", {})
            }

        except Exception as e:
            logger.error(f"Error in RAG workflow: {str(e)}")
            return {
                "query": str(initial_state["query"]),
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "citations": [],
                "context": "",
                "metadata": {
                    "error": str(e),
                    "workflow_failed": True
                }
            }

    async def process_query_stream(self, query: str) -> AsyncIterator[str]:
        """
        Process a user query through the RAG workflow with streaming response.

        Args:
            query: User query string

        Yields:
            str: Streaming response chunks with citations
        """
        logger.info(f"Processing query through RAG workflow with streaming: '{query[:50]}...'")

        try:
            # Step 1: Retrieve context
            logger.info("Step 1: Retrieving context...")
            context, citations = self.rag_service.retrieve_context(
                query=query,
                limit=self.config.get('qdrant', {}).get('search', {}).get('default_limit', 5),
                score_threshold=self.config.get('qdrant', {}).get('search', {}).get('default_score_threshold', 0.0),
                include_citations=self.rag_config.get('generation', {}).get('include_sources', True)
            )

            # Step 2: Create generation prompt
            logger.info("Step 2: Creating generation prompt...")
            prompt = self._create_generation_prompt(query, context)

            # Step 3: Stream LLM response
            logger.info("Step 3: Streaming LLM response...")
            messages = cast(List[BaseMessage], [HumanMessage(content=prompt)])

            response_chunks = []
            async for chunk in self.llm._astream(messages):
                if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
                    content = chunk.message.content
                    if isinstance(content, str):
                        response_chunks.append(content)
                        yield content

            # Step 4: Add citations if enabled
            if self.rag_config.get('generation', {}).get('include_sources', True) and citations:
                logger.info("Step 4: Adding citations...")
                # Generate just the citations section
                citation_text = "\n\n**Sources:**\n"
                for citation in citations:
                    citation_id = citation.get('id', 'Unknown')
                    score = citation.get('score', 0.0)
                    movie_title = citation.get('movie_title', 'Unknown Movie')
                    review_title = citation.get('review_title', '')
                    year = citation.get('year', '')
                    genre = citation.get('genre', '')

                    citation_line = f"{citation_id}. **{movie_title}**"
                    if year:
                        citation_line += f" ({year})"
                    if genre:
                        citation_line += f" - {genre}"
                    if review_title:
                        citation_line += f"\n   Review: \"{review_title}\""
                    citation_line += f" (Relevance: {score:.2f})\n"
                    citation_text += citation_line

                yield citation_text

            # Log completion metadata
            full_response = ''.join(response_chunks)
            logger.info(f"RAG streaming completed - Retrieved docs: {len(citations)}, " +
                       f"Context length: {len(context)}, " +
                       f"Response length: {len(full_response)}, " +
                       f"Citations: {len(citations) > 0}")

        except Exception as e:
            logger.error(f"Error in RAG streaming workflow: {str(e)}")
            yield f"I apologize, but I encountered an error processing your query: {str(e)}"

    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow configuration.

        Returns:
            Dictionary with workflow information
        """
        return {
            "workflow_type": "RAG with LangGraph",
            "nodes": ["retrieve", "generate", "respond"],
            "rag_service": {
                "max_context_length": self.rag_service.max_context_length,
                "source_format": self.rag_service.source_format,
                "include_sources": self.rag_service.include_sources
            },
            "llm": {
                "type": "GeminiChat",
                "api_url": self.llm.api_url,
                "model_name": self.model_name
            }
        }
