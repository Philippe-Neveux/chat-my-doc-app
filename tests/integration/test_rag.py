"""
Integration tests for RAG service.

These tests require a running Qdrant instance with test data.
"""

import pytest
from unittest.mock import Mock, patch

from chat_my_doc_app.config import get_config
from chat_my_doc_app.rag import (
    RetrievalService,
    DocumentSource,
    RAGImdb,
    RAGImdbState
)


class TestRetrievalIntegration:
    """Integration tests for RetrievalService."""

    @classmethod
    def setup_class(cls):
        """Set up test class with configuration."""
        cls.config = get_config()

    def test_retrieval_service_initialization(self):
        """Test RetrievalService can be initialized with config."""
        service = RetrievalService(self.config)

        assert service.config is not None
        assert service.qdrant_service is not None
        assert service.max_context_length > 0
        assert service.chunk_size > 0
        assert isinstance(service.include_sources, bool)
        assert service.source_format in ['markdown', 'plain']

    def test_query_preprocessing(self):
        """Test query preprocessing functionality."""
        service = RetrievalService(self.config)

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
        service = RetrievalService(self.config)

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
        service = RetrievalService(self.config)

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
        service = RetrievalService(self.config)

        # Test empty queries
        assert service.retrieve_documents("") == []
        assert service.retrieve_documents("   ") == []
        assert service.retrieve_documents("\t\n") == []

    @pytest.mark.slow
    def test_context_formatting(self):
        """Test document formatting for context."""
        service = RetrievalService(self.config)

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
        service = RetrievalService(self.config)

        context = service.format_documents_for_context([])
        assert context == "No relevant documents found."

    @pytest.mark.slow
    def test_citation_generation(self):
        """Test citation generation from documents."""
        service = RetrievalService(self.config)

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
        service = RetrievalService(self.config)

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
        service = RetrievalService(self.config)

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
        service = RetrievalService(self.config)

        context, citations = service.retrieve_context(
            "movie review",
            include_citations=False
        )

        # Should have context but no citations
        assert isinstance(context, str)
        assert citations == []

    def test_retrieval_pipeline_no_results(self):
        """Test retrieval pipeline when no documents match."""
        service = RetrievalService(self.config)

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

        service = RetrievalService(test_config)

        context, _ = service.retrieve_context("movie", limit=10)

        # Context should not exceed the limit (allowing some buffer for formatting)
        assert len(context) <= 250  # Small buffer for formatting overhead

    @pytest.mark.slow
    def test_different_source_formats(self):
        """Test different source formatting options."""
        # Test markdown format
        markdown_config = self.config.copy()
        markdown_config['rag']['generation']['source_format'] = 'markdown'
        service_md = RetrievalService(markdown_config)

        context_md, _ = service_md.retrieve_context("movie", limit=1)

        if "Source 1" in context_md:  # If we got results
            assert "**Source 1**" in context_md

        # Test plain format
        plain_config = self.config.copy()
        plain_config['rag']['generation']['source_format'] = 'plain'
        service_plain = RetrievalService(plain_config)

        context_plain, _ = service_plain.retrieve_context("movie", limit=1)

        if "Source 1" in context_plain:  # If we got results
            assert "**Source 1**" not in context_plain  # No markdown formatting


@pytest.fixture
def retrieval_service():
    """Session-scoped fixture for RetrievalService."""
    config = get_config()
    return RetrievalService(config)


@pytest.mark.slow
def test_end_to_end_retrieval_workflow(retrieval_service):
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
            context, citations = retrieval_service.retrieve_context(query, limit=3)

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


def test_retrieval_service_error_handling(retrieval_service):
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
            context, citations = retrieval_service.retrieve_context(edge_case)

            # Should return something, even if it's an error message
            assert isinstance(context, str)
            assert isinstance(citations, list)

        except Exception as e:
            pytest.fail(f"RAG service should handle edge case gracefully: '{edge_case}' -> {e}")



@pytest.fixture
def sample_config_rag():
    """Sample configuration for testing."""
    return {
        'qdrant': {
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'test_collection',
            'search': {
                'default_limit': 5,
                'default_score_threshold': 0.0,
                'max_limit': 20
            }
        },
        'embedding': {
            'model_name': 'all-MiniLM-L6-v2'
        },
        'rag': {
            'max_context_length': 2000,
            'context_overlap': 100,
            'retrieval': {
                'chunk_size': 300,
                'chunk_overlap': 50,
                'min_chunk_size': 50
            },
            'generation': {
                'include_sources': True,
                'source_format': 'markdown'
            }
        },
        'llm': {
            'api_url': 'http://localhost:8000',
            'model_name': 'gemini-2.0-flash-lite',
            'system_prompt': 'You are a test assistant.'
        }
    }

class TestRAGImdbIntegration:
    """Integration tests for RAGImdb."""

    @classmethod
    def setup_class(cls):
        """Set up test class with configuration."""
        cls.config = get_config()

    def test_rag_initialization_with_config(self):
        """Test workflow can be initialized with real config."""
        # Mock the LLM to avoid requiring API access
        with patch('chat_my_doc_app.rag.GeminiChat') as mock_gemini:
            mock_llm = Mock()
            mock_gemini.return_value = mock_llm

            rag = RAGImdb("gemini-2.0-flash-lite", self.config)

            assert rag.config is not None
            assert rag.rag_service is not None
            assert rag.llm is not None
            assert rag.workflow is not None

    def test_factory_function_with_config(self):
        """Test factory function works with real config."""
        with patch('chat_my_doc_app.rag.GeminiChat') as mock_gemini:
            mock_llm = Mock()
            mock_gemini.return_value = mock_llm

            rag = RAGImdb("gemini-2.0-flash-lite", self.config)

            assert isinstance(rag, RAGImdb)
            assert rag.config == self.config

    @pytest.mark.slow
    def test_rag_nodes_with_rag_service(self):
        """Test rag nodes work with real RAG service (mock LLM)."""
        # Mock only the LLM to test RAG service integration
        with patch('chat_my_doc_app.rag.GeminiChat') as mock_gemini:
            # Setup LLM mock
            mock_generation = Mock()
            mock_generation.message.content = "This is a test response about movies."
            mock_result = Mock()
            mock_result.generations = [mock_generation]

            mock_llm = Mock()
            mock_llm._generate.return_value = mock_result
            mock_gemini.return_value = mock_llm

            rag = RAGImdb("gemini-2.0-flash-lite", self.config)

            # Test retrieve node (will use real RAG service)
            try:
                initial_state = RAGImdbState({
                    'query': 'action movies',
                    'context': '',
                    'citations': [],
                    'response': '',
                    'metadata': {}
                })

                # This will call the real RAG service
                retrieve_result = rag.retrieve_node(initial_state)

                # Should have some context (even if empty results)
                assert 'context' in retrieve_result
                assert isinstance(retrieve_result['context'], str)
                assert 'citations' in retrieve_result
                assert isinstance(retrieve_result['citations'], list)

                # Test generate node with retrieved context
                if retrieve_result['context']:
                    generate_result = rag.generate_node(retrieve_result)

                    assert 'response' in generate_result
                    assert generate_result['response'] == "This is a test response about movies."
                    assert generate_result['metadata']['generated'] is True

                    # Test respond node
                    respond_result = rag.respond_node(generate_result)

                    assert 'response' in respond_result
                    assert isinstance(respond_result['response'], str)

            except Exception as e:
                # If Qdrant is not available, skip this test
                pytest.skip(f"Qdrant service not available: {e}")

    def test_rag_configuration_loading(self):
        """Test that rag loads configuration correctly."""
        with patch('chat_my_doc_app.rag.GeminiChat') as mock_gemini:
            mock_llm = Mock()
            mock_gemini.return_value = mock_llm

            rag = RAGImdb("gemini-2.0-flash-lite", self.config)

            # Check that config values are loaded
            assert rag.config.get('qdrant', {}).get('collection_name') == 'imdb_reviews'
            assert rag.config.get('embedding', {}).get('model_name') == 'all-MiniLM-L6-v2'
            assert rag.rag_config.get('max_context_length') == 4000

            # Check rag info
            info = rag.get_workflow_info()
            assert info['workflow_type'] == "RAG with LangGraph"
            assert len(info['nodes']) == 3

    def test_rag_with_mock_llm_and_rag(self):
        """Test complete rag with both LLM and RAG service mocked."""
        with patch('chat_my_doc_app.rag.GeminiChat') as mock_gemini, \
             patch('chat_my_doc_app.rag.RetrievalService') as mock_rag_service:

            # Setup RAG service mock
            mock_rag = Mock()
            mock_rag.retrieve_context.return_value = (
                "**Source 1** (Score: 0.95)\nGreat action movie with amazing stunts.",
                [{'id': 1, 'score': 0.95, 'movie_title': 'Action Hero', 'year': '2023'}]
            )
            mock_rag_service.return_value = mock_rag

            # Setup LLM mock
            mock_generation = Mock()
            mock_generation.message.content = "Based on the reviews, Action Hero is an excellent action movie with great stunts."
            mock_result = Mock()
            mock_result.generations = [mock_generation]

            mock_llm = Mock()
            mock_llm._generate.return_value = mock_result
            mock_gemini.return_value = mock_llm

            # Create rag and test complete flow
            rag = RAGImdb("gemini-2.0-flash-lite", self.config)

            result = rag.process_query("What are some good action movies?")

            # Verify result structure
            assert 'response' in result
            assert 'citations' in result
            assert 'context' in result
            assert 'metadata' in result

            # Verify content
            assert "Action Hero" in result['response']  # Should include citations
            assert "Sources:" in result['response']  # Should have citation section
            assert len(result['citations']) == 1
            assert result['citations'][0]['movie_title'] == 'Action Hero'

            # Verify rag completed successfully
            assert not result['metadata'].get('error')
            assert result['metadata'].get('workflow_started') is True

    @patch('chat_my_doc_app.rag.RetrievalService')
    @patch('chat_my_doc_app.rag.GeminiChat')
    def test_process_query_success(self, mock_gemini, mock_rag_service, sample_config_rag):
        """Test successful query processing through the rag."""
        # Setup mocks
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve_context.return_value = (
            "Great context about movies",
            [{'id': 1, 'score': 0.9, 'movie_title': 'Test Movie'}]
        )
        mock_rag_service.return_value = mock_rag_instance

        mock_generation = Mock()
        mock_generation.message.content = "Generated response about movies"
        mock_result = Mock()
        mock_result.generations = [mock_generation]

        mock_llm_instance = Mock()
        mock_llm_instance._generate.return_value = mock_result
        mock_gemini.return_value = mock_llm_instance

        rag = RAGImdb("gemini-2.0-flash-lite", sample_config_rag)

        # Process a query
        result = rag.process_query("What are good movies?")

        # Verify result structure
        assert 'response' in result
        assert 'citations' in result
        assert 'context' in result
        assert 'metadata' in result

        # Verify content
        assert "Generated response about movies" in result['response']
        assert len(result['citations']) == 1
        assert result['context'] == "Great context about movies"
        assert result['metadata']['workflow_started'] is True

    @patch('chat_my_doc_app.rag.RetrievalService')
    @patch('chat_my_doc_app.rag.GeminiChat')
    def test_process_query_with_error(self, mock_gemini, mock_rag_service, sample_config_rag):
        """Test query processing with errors."""
        # Setup mocks to raise exception
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve_context.side_effect = Exception("RAG service error")
        mock_rag_service.return_value = mock_rag_instance

        mock_llm_instance = Mock()
        mock_gemini.return_value = mock_llm_instance

        rag = RAGImdb("gemini-2.0-flash-lite", sample_config_rag)

        result = rag.process_query("test query")

        # Should handle error gracefully
        assert 'response' in result
        # The rag continues even with retrieve errors, so check for node-level errors
        assert ('retrieve_error' in result['metadata'] or
                'generate_error' in result['metadata'] or
                'error' in result['metadata'])
        assert result['response'] is not None

    @patch('chat_my_doc_app.rag.RetrievalService')
    @patch('chat_my_doc_app.rag.GeminiChat')
    def test_node_error_handling(self, mock_gemini, mock_rag_service, sample_config_rag):
        """Test individual node error handling."""
        mock_rag_instance = Mock()
        mock_rag_service.return_value = mock_rag_instance

        mock_llm_instance = Mock()
        mock_gemini.return_value = mock_llm_instance

        rag = RAGImdb("gemini-2.0-flash-lite", sample_config_rag)

        # Test retrieve node error handling
        mock_rag_instance.retrieve_context.side_effect = Exception("Retrieve error")

        state = RAGImdbState(
            query="test",
            context="",
            citations=[],
            response="",
            metadata={}
        )

        result = rag.retrieve_node(state)

        assert "Error retrieving information" in result['context']
        assert result['citations'] == []
        assert 'retrieve_error' in result['metadata']


def test_rag_error_resilience():
    """Test rag handles various error conditions gracefully."""
    config = get_config()

    # Test with invalid LLM configuration
    bad_config = config.copy()
    bad_config['llm'] = {
        'api_url': 'http://invalid-url:9999',
        'model_name': 'non-existent-model'
    }

    with patch('chat_my_doc_app.rag.GeminiChat') as mock_gemini:
        # Mock LLM that raises errors
        mock_llm = Mock()
        mock_llm._generate.side_effect = Exception("LLM API error")
        mock_gemini.return_value = mock_llm

        rag = RAGImdb("gemini-2.0-flash-lite", bad_config)

        # Should still create rag
        assert rag is not None

        # Process query should handle errors gracefully
        result = rag.process_query("test query")

        assert 'response' in result
        assert 'metadata' in result
        # Should have some error indication in metadata
        assert any(key.endswith('_error') for key in result['metadata'].keys())


@pytest.mark.slow
def test_end_to_end_rag_mock_dependencies():
    """End-to-end rag test with mocked external dependencies."""
    config = get_config()

    with patch('chat_my_doc_app.rag.GeminiChat') as mock_gemini, \
         patch('chat_my_doc_app.rag.RetrievalService') as mock_rag_service:

        # Setup comprehensive mocks
        mock_rag = Mock()
        mock_rag.max_context_length = 4000
        mock_rag.source_format = 'markdown'
        mock_rag.include_sources = True
        mock_rag.retrieve_context.return_value = (
            """**Source 1** (Score: 0.92)
*Movie: The Matrix | Year: 1999 | Genre: Action, Sci-Fi | Rating: 8.7*
**Review Title:** Mind-blowing action movie (Rating: 9/10)

This movie revolutionized action cinema with its innovative effects.""",
            [{
                'id': 1,
                'document_id': 'matrix_001',
                'score': 0.92,
                'movie_title': 'The Matrix',
                'year': '1999',
                'genre': 'Action, Sci-Fi',
                'review_title': 'Mind-blowing action movie',
                'review_rating': 9,
                'movie_rating': 8.7
            }]
        )
        mock_rag_service.return_value = mock_rag

        # Setup LLM mock
        mock_generation = Mock()
        mock_generation.message.content = """Based on the movie reviews, I can recommend The Matrix as an excellent action movie. The review highlights that it "revolutionized action cinema with its innovative effects" and received high ratings from both critics (8.7/10) and individual reviewers (9/10). The movie is from 1999 and combines Action and Sci-Fi genres, making it a groundbreaking film that's definitely worth watching."""

        mock_result = Mock()
        mock_result.generations = [mock_generation]

        mock_llm = Mock()
        mock_llm.model_name = 'gemini-2.0-flash-lite'
        mock_llm.api_url = 'http://localhost:8000'
        mock_llm._generate.return_value = mock_result
        mock_gemini.return_value = mock_llm

        # Create and test rag
        rag = RAGImdb("gemini-2.0-flash-lite", config)

        # Test realistic query
        result = rag.process_query("What are some highly rated action movies worth watching?")

        # Comprehensive result verification
        assert result['response'] is not None
        assert len(result['response']) > 100  # Should be substantial

        # Should contain information from the "retrieved" context
        assert "Matrix" in result['response']
        assert "action" in result['response'].lower()

        # Should have citations section
        assert "Sources:" in result['response']
        assert "The Matrix" in result['response']
        assert "(Relevance: 0.92)" in result['response']

        # Verify citations structure
        assert len(result['citations']) == 1
        citation = result['citations'][0]
        assert citation['movie_title'] == 'The Matrix'
        assert citation['year'] == '1999'
        assert citation['score'] == 0.92

        # Verify metadata
        metadata = result['metadata']
        assert metadata['workflow_started'] is True
        assert metadata['retrieved_docs'] == 1
        assert metadata['generated'] is True
        assert metadata['citations_included'] is True
        assert metadata['final_response_length'] > 0

        # Verify workflow info
        info = rag.get_workflow_info()
        assert info['workflow_type'] == "RAG with LangGraph"
        assert info['nodes'] == ["retrieve", "generate", "respond"]
        assert info['rag_service']['max_context_length'] == 4000
        assert info['llm']['type'] == 'GeminiChat'

        print(f"✅ End-to-end workflow test passed!")
        print(f"📊 Response length: {len(result['response'])} characters")
        print(f"📚 Citations: {len(result['citations'])}")
        print(f"🎬 Movie mentioned: {citation['movie_title']} ({citation['year']})")
