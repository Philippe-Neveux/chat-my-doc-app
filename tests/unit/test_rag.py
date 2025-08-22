"""
Unit tests for RAG service.

These tests focus on individual components of the RAG service without
requiring external dependencies like Qdrant.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from chat_my_doc_app.llms import GeminiChat
from chat_my_doc_app.rag import DocumentSource, RAGImdb, RAGImdbState, RetrievalService


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


class TestRetrievalService:
    """Test the RetrievalService class."""

    @patch('chat_my_doc_app.rag.QdrantService')
    def test_rag_service_initialization(self, mock_qdrant_class, sample_config):
        """Test RetrievalService initialization."""
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant

        service = RetrievalService(sample_config)

        assert service.config == sample_config
        assert service.max_context_length == 2000
        assert service.chunk_size == 300
        assert service.include_sources is True
        assert service.source_format == 'markdown'

        mock_qdrant_class.assert_called_once_with(sample_config)

    def test_preprocess_query(self, sample_config):
        """Test query preprocessing."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RetrievalService(sample_config)

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

        service = RetrievalService(sample_config)

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

        service = RetrievalService(sample_config)

        # Test empty query
        docs = service.retrieve_documents("")
        assert docs == []

        # Test whitespace-only query
        docs = service.retrieve_documents("   ")
        assert docs == []

    def test_format_documents_for_context(self, sample_config, sample_documents):
        """Test document formatting for context."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RetrievalService(sample_config)

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
            service = RetrievalService(sample_config)

        context = service.format_documents_for_context(sample_documents)

        assert "Source 1" in context
        assert "**Source 1**" not in context  # No markdown formatting
        assert "Score: 0.950" in context

    def test_format_documents_empty_list(self, sample_config):
        """Test formatting with empty document list."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RetrievalService(sample_config)

        context = service.format_documents_for_context([])
        assert context == "No relevant documents found."

    def test_format_documents_context_length_limit(self, sample_config):
        """Test context length limiting."""
        # Create a document with very long content
        long_content = "A" * 1500  # Longer than our test max_context_length of 2000
        long_doc = DocumentSource("long_doc", long_content, {}, 0.9)

        sample_config['rag']['max_context_length'] = 500  # Very short limit

        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RetrievalService(sample_config)

        context = service.format_documents_for_context([long_doc])

        # Should be truncated
        assert len(context) <= 500
        assert "..." in context or "Truncated" in context

    def test_generate_citations(self, sample_config, sample_documents):
        """Test citation generation."""
        with patch('chat_my_doc_app.rag.QdrantService'):
            service = RetrievalService(sample_config)

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

        service = RetrievalService(sample_config)

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

        service = RetrievalService(sample_config)

        context, citations = service.retrieve_context("non-existent topic")

        assert context == "No relevant information found for your query."
        assert citations == []




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


@pytest.fixture
def sample_state():
    """Sample workflow state for testing."""
    return RAGImdbState(
        query="What are good action movies?",
        context="**Source 1** (Score: 0.95)\nGreat action movie with amazing effects.",
        citations=[{
            'id': 1,
            'document_id': 'doc1',
            'score': 0.95,
            'movie_title': 'Action Hero',
            'year': '2023',
            'genre': 'Action'
        }],
        response="Based on the reviews, Action Hero is a great action movie.",
        metadata={'test': True}
    )


class TestRAGImdb:
    """Test the TestRAGImdb class."""

    @patch('chat_my_doc_app.rag.RetrievalService')
    def test_rag_initialization(self, mock_rag_service, sample_config_rag):
        """Test RAGImdb initialization."""
        mock_rag_instance = Mock()
        mock_rag_service.return_value = mock_rag_instance

        # Create a mock GeminiChat instance
        mock_llm_instance = Mock(spec=GeminiChat)
        mock_llm_instance.system_prompt = "You are a helpful assistant."

        rag = RAGImdb(mock_llm_instance, sample_config_rag)

        assert rag.config == sample_config_rag
        assert rag.rag_service == mock_rag_instance
        assert rag.llm == mock_llm_instance
        assert rag.workflow is not None

        # Verify RetrievalService was initialized correctly
        mock_rag_service.assert_called_once_with(sample_config_rag)

    @patch('chat_my_doc_app.rag.RetrievalService')
    def test_retrieve_node(self, mock_rag_service, sample_config_rag):
        """Test the retrieve node functionality."""
        # Setup mocks
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve_context.return_value = (
            "Test context with movie reviews",
            [{'id': 1, 'score': 0.9, 'movie_title': 'Test Movie'}]
        )
        mock_rag_service.return_value = mock_rag_instance

        # Create a mock GeminiChat instance
        mock_llm_instance = Mock(spec=GeminiChat)
        mock_llm_instance.system_prompt = "You are a helpful assistant."

        rag = RAGImdb(mock_llm_instance, sample_config_rag)

        # Test retrieve node
        initial_state = RAGImdbState(
            query="test query",
            context="",
            citations=[],
            response="",
            metadata={}
        )

        result = rag.retrieve_node(initial_state)

        # Verify the retrieve_context was called correctly
        mock_rag_instance.retrieve_context.assert_called_once_with(
            query="test query",
            limit=5,
            score_threshold=0.0,
            include_citations=True
        )

        # Verify state updates
        assert result['query'] == "test query"
        assert result['context'] == "Test context with movie reviews"
        assert len(result['citations']) == 1
        assert result['citations'][0]['movie_title'] == 'Test Movie'
        assert result['metadata']['retrieved_docs'] == 1
        assert result['metadata']['context_length'] == len("Test context with movie reviews")

    @patch('chat_my_doc_app.rag.RetrievalService')
    def test_generate_node(self, mock_rag_service, sample_config_rag):
        """Test the generate node functionality."""
        # Setup mocks
        mock_rag_instance = Mock()
        mock_rag_service.return_value = mock_rag_instance

        # Mock LLM response
        mock_generation = Mock()
        mock_generation.message.content = "This is a generated response about movies."
        mock_result = Mock()
        mock_result.generations = [mock_generation]

        # Create a mock GeminiChat instance
        mock_llm_instance = Mock(spec=GeminiChat)
        mock_llm_instance.system_prompt = "You are a helpful assistant."
        mock_llm_instance._generate.return_value = mock_result

        rag = RAGImdb(mock_llm_instance, sample_config_rag)

        # Test generate node
        state_with_context = RAGImdbState(
            query="What are good movies?",
            context="Context about great movies",
            citations=[],
            response="",
            metadata={}
        )

        result = rag.generate_node(state_with_context)

        # Verify LLM was called
        mock_llm_instance._generate.assert_called_once()

        # Verify state updates
        assert result['query'] == "What are good movies?"
        assert result['context'] == "Context about great movies"
        assert result['response'] == "This is a generated response about movies."
        assert result['metadata']['generated'] is True
        assert result['metadata']['response_length'] == len("This is a generated response about movies.")

    @patch('chat_my_doc_app.rag.RetrievalService')
    def test_respond_node(self, mock_rag_service, sample_config_rag, sample_state):
        """Test the respond node functionality."""
        mock_rag_instance = Mock()
        mock_rag_service.return_value = mock_rag_instance

        # Create a mock GeminiChat instance
        mock_llm_instance = Mock(spec=GeminiChat)
        mock_llm_instance.system_prompt = "You are a helpful assistant."

        rag = RAGImdb(mock_llm_instance, sample_config_rag)

        result = rag.respond_node(sample_state)

        # Should include citations in response since include_sources is True
        assert result['query'] == sample_state['query']
        assert result['response'] != sample_state['response']  # Should be modified with citations
        assert "Sources:" in result['response']
        assert "Action Hero" in result['response']
        assert result['metadata']['citations_included'] is True

    @patch('chat_my_doc_app.rag.RetrievalService')
    def test_respond_node_no_citations(self, mock_rag_service, sample_config_rag):
        """Test respond node with citations disabled."""
        # Modify config to disable citations
        config_no_citations = sample_config_rag.copy()
        config_no_citations['rag']['generation']['include_sources'] = False

        mock_rag_instance = Mock()
        mock_rag_service.return_value = mock_rag_instance

        # Create a mock GeminiChat instance
        mock_llm_instance = Mock(spec=GeminiChat)
        mock_llm_instance.system_prompt = "You are a helpful assistant."

        rag = RAGImdb(mock_llm_instance, config_no_citations)

        state = RAGImdbState(
            query="test",
            context="context",
            citations=[{'id': 1, 'movie_title': 'Movie'}],
            response="Original response",
            metadata={}
        )

        result = rag.respond_node(state)

        # Should NOT include citations
        assert result['response'] == "Original response"
        assert "Sources:" not in result['response']

    def test_create_generation_prompt(self, sample_config_rag):
        """Test generation prompt creation."""
        with patch('chat_my_doc_app.rag.RetrievalService'):
            # Create a mock GeminiChat instance
            mock_llm_instance = Mock(spec=GeminiChat)
            mock_llm_instance.system_prompt = "You are a helpful assistant."

            rag = RAGImdb(mock_llm_instance, sample_config_rag)

            query = "What are the best movies?"
            context = "Review 1: Great movie with excellent acting."

            prompt = rag._create_generation_prompt(query, context)

            assert query in prompt
            assert context in prompt
            assert "Based on the following context" in prompt
            assert "User Question:" in prompt

    def test_add_citations_to_response(self, sample_config_rag):
        """Test citation formatting."""
        with patch('chat_my_doc_app.rag.RetrievalService'):
            # Create a mock GeminiChat instance
            mock_llm_instance = Mock(spec=GeminiChat)
            mock_llm_instance.system_prompt = "You are a helpful assistant."

            rag = RAGImdb(mock_llm_instance, sample_config_rag)

            response = "This is a great movie."
            citations = [
                {
                    'id': 1,
                    'score': 0.95,
                    'movie_title': 'Action Hero',
                    'year': '2023',
                    'genre': 'Action',
                    'review_title': 'Amazing film'
                }
            ]

            result = rag._add_citations_to_response(response, citations)

            assert "This is a great movie." in result
            assert "**Sources:**" in result
            assert "1. **Action Hero**" in result
            assert "(2023)" in result
            assert "Action" in result
            assert "Amazing film" in result
            assert "(Relevance: 0.95)" in result

    def test_add_citations_empty_list(self, sample_config_rag):
        """Test citation formatting with empty citations."""
        with patch('chat_my_doc_app.rag.RetrievalService'):
            # Create a mock GeminiChat instance
            mock_llm_instance = Mock(spec=GeminiChat)
            mock_llm_instance.system_prompt = "You are a helpful assistant."

            rag = RAGImdb(mock_llm_instance, sample_config_rag)

            response = "This is a response."
            citations = []

            result = rag._add_citations_to_response(response, citations)

            # Should return original response unchanged
            assert result == response
            assert "Sources:" not in result

    @patch('chat_my_doc_app.rag.RetrievalService')
    def test_get_rag_info(self, mock_rag_service, sample_config_rag):
        """Test rag information retrieval."""
        # Setup mocks with attributes
        mock_rag_instance = Mock()
        mock_rag_instance.max_context_length = 2000
        mock_rag_instance.source_format = 'markdown'
        mock_rag_instance.include_sources = True
        mock_rag_service.return_value = mock_rag_instance

        # Create a mock GeminiChat instance
        mock_llm_instance = Mock(spec=GeminiChat)
        mock_llm_instance.system_prompt = "You are a helpful assistant."
        mock_llm_instance.model_name = 'gemini-2.0-flash-lite'
        mock_llm_instance.api_url = 'http://localhost:8000'
        mock_llm_instance._llm_type = 'GeminiChat'

        rag = RAGImdb(mock_llm_instance, sample_config_rag)

        info = rag.get_workflow_info()

        assert info['workflow_type'] == "RAG with LangGraph"
        assert info['nodes'] == ["retrieve", "generate", "respond"]
        assert info['rag_service']['max_context_length'] == 2000
        assert info['rag_service']['source_format'] == 'markdown'
        assert info['llm']['type'] == 'GeminiChat'
        assert info['llm']['model_name'] == 'gemini-2.0-flash-lite'

    def test_factory_function(self, sample_config_rag):
        """Test the create_rag_workflow factory function."""
        with patch('chat_my_doc_app.rag.RetrievalService'):
            # Create a mock GeminiChat instance
            mock_llm_instance = Mock(spec=GeminiChat)
            mock_llm_instance.system_prompt = "You are a helpful assistant."

            rag = RAGImdb(mock_llm_instance, sample_config_rag)

            assert isinstance(rag, RAGImdb)
            assert rag.config == sample_config_rag
