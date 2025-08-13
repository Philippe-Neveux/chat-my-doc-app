"""
Unit tests for RAG service.

These tests focus on individual components of the RAG service without
requiring external dependencies like Qdrant.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from chat_my_doc_app.rag import (
    RAGService,
    DocumentSource
)


@pytest.fixture
def sample_config():
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
        }
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        DocumentSource(
            doc_id="doc1",
            content="This is a great movie with excellent acting and beautiful cinematography.",
            metadata={"title": "Movie A", "rating": 5, "sentiment": "positive"},
            score=0.95
        ),
        DocumentSource(
            doc_id="doc2",
            content="The plot was confusing and the characters were poorly developed.",
            metadata={"title": "Movie B", "rating": 2, "sentiment": "negative"},
            score=0.75
        ),
        DocumentSource(
            doc_id="doc3",
            content="A masterpiece of cinema with outstanding performances by all actors.",
            metadata={"title": "Movie C", "rating": 5, "sentiment": "positive"},
            score=0.88
        )
    ]


class TestDocumentSource:
    """Test the DocumentSource class."""

    def test_document_source_creation(self):
        """Test DocumentSource object creation."""
        doc = DocumentSource(
            doc_id="test123",
            content="Test content",
            metadata={"key": "value"},
            score=0.85
        )

        assert doc.id == "test123"
        assert doc.content == "Test content"
        assert doc.metadata == {"key": "value"}
        assert doc.score == 0.85
        assert doc.citation_id is None

    def test_document_source_repr(self):
        """Test DocumentSource string representation."""
        doc = DocumentSource("doc1", "content", {}, 0.123)
        assert "doc1" in repr(doc)
        assert "0.123" in repr(doc)


class TestRAGService:
    """Test the RAGService class."""

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_rag_service_initialization(self, mock_qdrant_class, sample_config):
        """Test RAGService initialization."""
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant

        service = RAGService(sample_config)

        assert service.config == sample_config
        assert service.max_context_length == 2000
        assert service.chunk_size == 300
        assert service.include_sources is True
        assert service.source_format == 'markdown'

        mock_qdrant_class.assert_called_once_with(sample_config)

    def test_preprocess_query(self, sample_config):
        """Test query preprocessing."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RAGService(sample_config)

        # Test basic cleaning
        assert service.preprocess_query("  hello world  ") == "hello world"

        # Test empty query
        assert service.preprocess_query("") == ""
        assert service.preprocess_query("   ") == ""

        # Test query preservation
        assert service.preprocess_query("great movies") == "great movies"

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_retrieve_documents(self, mock_qdrant_class, sample_config):
        """Test document retrieval."""
        # Mock Qdrant service
        mock_qdrant = Mock()
        mock_search_results = [
            ({'id': 'doc1', 'payload': {'review': 'Great movie!', 'rating': 5}}, 0.95),
            ({'id': 'doc2', 'payload': {'review': 'Terrible film.', 'rating': 1}}, 0.60)
        ]
        mock_qdrant.similarity_search.return_value = mock_search_results
        mock_qdrant_class.return_value = mock_qdrant

        service = RAGService(sample_config)

        # Test retrieval
        docs = service.retrieve_documents("great movies", limit=2, score_threshold=0.5)

        assert len(docs) == 2
        assert docs[0].id == 'doc1'
        assert docs[0].content == 'Great movie!'
        assert docs[0].score == 0.95
        assert docs[1].id == 'doc2'
        assert docs[1].content == 'Terrible film.'
        assert docs[1].score == 0.60

        # Verify Qdrant was called correctly
        mock_qdrant.similarity_search.assert_called_once_with(
            query="great movies",
            limit=2,
            score_threshold=0.5,
            metadata_filter=None
        )

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_retrieve_documents_empty_query(self, mock_qdrant_class, sample_config):
        """Test retrieval with empty query."""
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant

        service = RAGService(sample_config)

        # Test empty query
        docs = service.retrieve_documents("")
        assert docs == []

        # Test whitespace-only query
        docs = service.retrieve_documents("   ")
        assert docs == []

    def test_format_documents_for_context(self, sample_config, sample_documents):
        """Test document formatting for context."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RAGService(sample_config)

        # Test formatting with markdown
        context = service.format_documents_for_context(sample_documents)

        assert "**Source 1**" in context
        assert "Score: 0.950" in context
        assert "excellent acting" in context
        assert "**Source 2**" in context
        assert "poorly developed" in context

        # Check citation IDs were assigned
        assert sample_documents[0].citation_id == 1
        assert sample_documents[1].citation_id == 2
        assert sample_documents[2].citation_id == 3

    def test_format_documents_for_context_plain(self, sample_config, sample_documents):
        """Test document formatting with plain text format."""
        sample_config['rag']['generation']['source_format'] = 'plain'

        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RAGService(sample_config)

        context = service.format_documents_for_context(sample_documents)

        assert "Source 1" in context
        assert "**Source 1**" not in context  # No markdown formatting
        assert "Score: 0.950" in context

    def test_format_documents_empty_list(self, sample_config):
        """Test formatting with empty document list."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RAGService(sample_config)

        context = service.format_documents_for_context([])
        assert context == "No relevant documents found."

    def test_format_documents_context_length_limit(self, sample_config):
        """Test context length limiting."""
        # Create a document with very long content
        long_content = "A" * 1500  # Longer than our test max_context_length of 2000
        long_doc = DocumentSource("long_doc", long_content, {}, 0.9)

        sample_config['rag']['max_context_length'] = 500  # Very short limit

        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RAGService(sample_config)

        context = service.format_documents_for_context([long_doc])

        # Should be truncated
        assert len(context) <= 500
        assert "..." in context or "Truncated" in context

    def test_generate_citations(self, sample_config, sample_documents):
        """Test citation generation."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RAGService(sample_config)

        # Set citation IDs (normally done by format_documents_for_context)
        for i, doc in enumerate(sample_documents, 1):
            doc.citation_id = i

        citations = service.generate_citations(sample_documents)

        assert len(citations) == 3

        # Check first citation
        assert citations[0]['id'] == 1
        assert citations[0]['document_id'] == 'doc1'
        assert citations[0]['score'] == 0.95
        assert citations[0]['movie_title'] == 'Movie A'
        assert citations[0]['movie_rating'] == 5
        assert citations[0]['metadata']['sentiment'] == 'positive'

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_retrieve_context_complete_pipeline(self, mock_qdrant_class, sample_config):
        """Test the complete retrieve_context pipeline."""
        # Mock Qdrant service
        mock_qdrant = Mock()
        mock_search_results = [
            ({'id': 'doc1', 'payload': {'review': 'Great movie!', 'title': 'Test Movie'}}, 0.95)
        ]
        mock_qdrant.similarity_search.return_value = mock_search_results
        mock_qdrant_class.return_value = mock_qdrant

        service = RAGService(sample_config)

        # Test complete pipeline
        context, citations = service.retrieve_context("tell me about great movies")

        # Check context
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Great movie!" in context
        assert "**Source 1**" in context

        # Check citations
        assert len(citations) == 1
        assert citations[0]['id'] == 1
        assert citations[0]['document_id'] == 'doc1'
        assert citations[0]['movie_title'] == 'Test Movie'

        # Verify query preprocessing happened
        mock_qdrant.similarity_search.assert_called_once()
        call_args = mock_qdrant.similarity_search.call_args[1]
        assert call_args['query'] == 'tell me about great movies'  # Query preserved

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_retrieve_context_no_results(self, mock_qdrant_class, sample_config):
        """Test retrieve_context with no search results."""
        mock_qdrant = Mock()
        mock_qdrant.similarity_search.return_value = []
        mock_qdrant_class.return_value = mock_qdrant

        service = RAGService(sample_config)

        context, citations = service.retrieve_context("non-existent topic")

        assert context == "No relevant information found for your query."
        assert citations == []

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_get_retrieval_stats(self, mock_qdrant_class, sample_config):
        """Test retrieval statistics generation."""
        mock_qdrant = Mock()
        mock_search_results = [
            ({'id': 'doc1', 'payload': {'text': 'Text 1'}}, 0.95),
            ({'id': 'doc2', 'payload': {'text': 'Text 2'}}, 0.75),
            ({'id': 'doc3', 'payload': {'text': 'Text 3'}}, 0.55)
        ]
        mock_qdrant.similarity_search.return_value = mock_search_results
        mock_qdrant_class.return_value = mock_qdrant

        service = RAGService(sample_config)

        stats = service.get_retrieval_stats("test query")

        assert stats['query'] == 'test query'
        assert stats['processed_query'] == 'test query'
        assert stats['total_results'] == 3
        assert stats['avg_score'] == (0.95 + 0.75 + 0.55) / 3
        assert stats['score_range'] == (0.55, 0.95)
        assert stats['top_score'] == 0.95

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_get_retrieval_stats_no_results(self, mock_qdrant_class, sample_config):
        """Test retrieval statistics with no results."""
        mock_qdrant = Mock()
        mock_qdrant.similarity_search.return_value = []
        mock_qdrant_class.return_value = mock_qdrant

        service = RAGService(sample_config)

        stats = service.get_retrieval_stats("empty query")

        assert stats['total_results'] == 0
        assert stats['avg_score'] == 0.0
        assert stats['score_range'] == (0.0, 0.0)
