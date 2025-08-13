"""
Integration tests for RAG service.

These tests require a running Qdrant instance with test data.
"""

import pytest

from chat_my_doc_app.config import get_config
from chat_my_doc_app.rag import RAGService, DocumentSource


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for RAGService."""

    @classmethod
    def setup_class(cls):
        """Set up test class with configuration."""
        cls.config = get_config()

    def test_rag_service_initialization(self):
        """Test RAGService can be initialized with config."""
        service = RAGService(self.config)

        assert service.config is not None
        assert service.qdrant_service is not None
        assert service.max_context_length > 0
        assert service.chunk_size > 0
        assert isinstance(service.include_sources, bool)
        assert service.source_format in ['markdown', 'plain']

    def test_factory_function(self):
        """Test the factory function creates a service correctly."""
        service = RAGService(self.config)

        assert isinstance(service, RAGService)
        assert service.config == self.config

    def test_query_preprocessing(self):
        """Test query preprocessing functionality."""
        service = RAGService(self.config)

        # Test various query preprocessing scenarios
        test_cases = [
            ("  great movies  ", "great movies"),
            ("", ""),
            ("   ", ""),
            ("movie", "movie"),  # Short query preserved
        ]

        for input_query, expected_output in test_cases:
            result = service.preprocess_query(input_query)
            assert result == expected_output, f"Failed for input: '{input_query}'"

    @pytest.mark.slow
    def test_document_retrieval_basic(self):
        """Test basic document retrieval from Qdrant."""
        service = RAGService(self.config)

        # Test retrieval with a basic query
        documents = service.retrieve_documents("movie review", limit=3)

        assert isinstance(documents, list)
        assert len(documents) <= 3

        # Check document structure
        for doc in documents:
            assert isinstance(doc, DocumentSource)
            assert hasattr(doc, 'id')
            assert hasattr(doc, 'content')
            assert hasattr(doc, 'metadata')
            assert hasattr(doc, 'score')
            assert isinstance(doc.score, float)
            assert 0 <= doc.score <= 1

    @pytest.mark.slow
    def test_document_retrieval_with_filters(self):
        """Test document retrieval with score threshold."""
        service = RAGService(self.config)

        # Test with score threshold
        documents_high_threshold = service.retrieve_documents(
            "excellent movie",
            limit=5,
            score_threshold=0.2
        )

        documents_low_threshold = service.retrieve_documents(
            "excellent movie",
            limit=5,
            score_threshold=0.0
        )

        # High threshold should return fewer or equal results
        assert len(documents_high_threshold) <= len(documents_low_threshold)

        # All high threshold results should meet the threshold
        for doc in documents_high_threshold:
            assert doc.score >= 0.2

    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        service = RAGService(self.config)

        # Test empty queries
        assert service.retrieve_documents("") == []
        assert service.retrieve_documents("   ") == []
        assert service.retrieve_documents("\t\n") == []

    @pytest.mark.slow
    def test_context_formatting(self):
        """Test document formatting for context."""
        service = RAGService(self.config)

        # Get some documents
        documents = service.retrieve_documents("movie", limit=2)

        if not documents:
            pytest.skip("No documents found for testing")

        # Test formatting
        context = service.format_documents_for_context(documents)

        assert isinstance(context, str)
        assert len(context) > 0

        # Check for expected formatting elements
        if service.source_format == 'markdown':
            assert "**Source 1**" in context
        else:
            assert "Source 1" in context

        assert "Score:" in context

        # Check citation IDs were assigned
        for i, doc in enumerate(documents, 1):
            assert doc.citation_id == i

    def test_context_formatting_empty_documents(self):
        """Test formatting with no documents."""
        service = RAGService(self.config)

        context = service.format_documents_for_context([])
        assert context == "No relevant documents found."

    @pytest.mark.slow
    def test_citation_generation(self):
        """Test citation generation from documents."""
        service = RAGService(self.config)

        # Get some documents and format them (to set citation_id)
        documents = service.retrieve_documents("movie", limit=2)

        if not documents:
            pytest.skip("No documents found for testing")

        # Format to set citation IDs
        service.format_documents_for_context(documents)

        # Generate citations
        citations = service.generate_citations(documents)

        assert isinstance(citations, list)
        assert len(citations) == len(documents)

        # Check citation structure
        for i, citation in enumerate(citations):
            assert 'id' in citation
            assert 'document_id' in citation
            assert 'score' in citation
            assert 'metadata' in citation
            assert citation['id'] == i + 1

    @pytest.mark.slow
    def test_complete_retrieval_pipeline(self):
        """Test the complete retrieve_context pipeline."""
        service = RAGService(self.config)

        # Test complete pipeline
        context, citations = service.retrieve_context("great movie with good acting")

        # Check context
        assert isinstance(context, str)
        assert len(context) > 0

        # Check citations
        assert isinstance(citations, list)

        if citations:  # If we got results
            # Check citation structure
            for citation in citations:
                assert 'id' in citation
                assert 'document_id' in citation
                assert 'score' in citation

            # Context should reference sources
            assert any(f"Source {i}" in context for i in range(1, len(citations) + 1))

    @pytest.mark.slow
    def test_retrieval_pipeline_with_parameters(self):
        """Test retrieval pipeline with custom parameters."""
        service = RAGService(self.config)

        # Test with custom parameters
        context, citations = service.retrieve_context(
            "tell me about excellent movies",
            limit=3,
            score_threshold=0.1,
            include_citations=True
        )

        # Should respect limit
        assert len(citations) <= 3

        # All citations should meet score threshold
        for citation in citations:
            assert citation['score'] >= 0.1

    @pytest.mark.slow
    def test_retrieval_pipeline_no_citations(self):
        """Test retrieval pipeline with citations disabled."""
        service = RAGService(self.config)

        context, citations = service.retrieve_context(
            "movie review",
            include_citations=False
        )

        # Should have context but no citations
        assert isinstance(context, str)
        assert citations == []

    def test_retrieval_pipeline_no_results(self):
        """Test retrieval pipeline when no documents match."""
        service = RAGService(self.config)

        # Use a very specific query unlikely to match
        context, citations = service.retrieve_context(
            "xyzzyx_nonexistent_query_12345",
            score_threshold=0.9  # Very high threshold
        )

        assert context == "No relevant information found for your query."
        assert citations == []

    @pytest.mark.slow
    def test_context_length_limits(self):
        """Test that context respects length limits."""
        # Create a config with very small context limit for testing
        test_config = self.config.copy()
        test_config['rag']['max_context_length'] = 200  # Very small

        service = RAGService(test_config)

        context, _ = service.retrieve_context("movie", limit=10)

        # Context should not exceed the limit (allowing some buffer for formatting)
        assert len(context) <= 250  # Small buffer for formatting overhead

    @pytest.mark.slow
    def test_different_source_formats(self):
        """Test different source formatting options."""
        # Test markdown format
        markdown_config = self.config.copy()
        markdown_config['rag']['generation']['source_format'] = 'markdown'
        service_md = RAGService(markdown_config)

        context_md, _ = service_md.retrieve_context("movie", limit=1)

        if "Source 1" in context_md:  # If we got results
            assert "**Source 1**" in context_md

        # Test plain format
        plain_config = self.config.copy()
        plain_config['rag']['generation']['source_format'] = 'plain'
        service_plain = RAGService(plain_config)

        context_plain, _ = service_plain.retrieve_context("movie", limit=1)

        if "Source 1" in context_plain:  # If we got results
            assert "**Source 1**" not in context_plain  # No markdown formatting


@pytest.fixture
def rag_service():
    """Session-scoped fixture for RAGService."""
    config = get_config()
    return RAGService(config)


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_rag_workflow(rag_service):
    """End-to-end test of the complete RAG workflow."""
    # Test realistic movie queries on your IMDB dataset

    test_queries = [
        "movies with great acting",
        "romantic comedy films",
        "action movies with good reviews",
        "worst movie ever made"
    ]

    for query in test_queries:
        try:
            # Test complete pipeline
            context, citations = rag_service.retrieve_context(query, limit=3)

            # Verify we got reasonable results
            assert isinstance(context, str)
            assert len(context) > 50  # Should have substantial content

            if citations:
                assert len(citations) <= 3

                # Citations should be sorted by relevance
                scores = [c['score'] for c in citations]
                assert scores == sorted(scores, reverse=True)

                # All citations should have required fields
                for citation in citations:
                    assert 'id' in citation
                    assert 'document_id' in citation
                    assert 'score' in citation
                    assert citation['score'] > 0

        except Exception as e:
            pytest.fail(f"RAG workflow failed for query '{query}': {e}")


@pytest.mark.integration
def test_rag_service_error_handling(rag_service):
    """Test RAG service error handling."""

    # Test with various edge cases that shouldn't crash
    edge_cases = [
        "",  # Empty string
        " " * 100,  # Long whitespace
        "a",  # Single character
        "!" * 50,  # Special characters
        "test" * 100,  # Very long query
    ]

    for edge_case in edge_cases:
        try:
            context, citations = rag_service.retrieve_context(edge_case)

            # Should return something, even if it's an error message
            assert isinstance(context, str)
            assert isinstance(citations, list)

        except Exception as e:
            pytest.fail(f"RAG service should handle edge case gracefully: '{edge_case}' -> {e}")
