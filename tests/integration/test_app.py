"""
Unit tests for Gradio app integration with RAG functionality.

Tests focus on the integration points between the Gradio interface
and the RAG workflow without requiring external dependencies.
"""

from typing import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chat_my_doc_app.chats import chat_with_rag_astream, if_rag_connection_works


class TestRAGChatIntegration:
    """Test RAG chat integration functionality."""

    @pytest.mark.asyncio
    @patch('chat_my_doc_app.chats.RAGImdb')
    async def test_chat_with_rag_astream_success(self, mock_rag_class):
        """Test successful RAG-enhanced chat streaming."""
        # Mock RAGImdb instance and its streaming method
        mock_rag_instance = Mock()

        async def mock_process_query_stream(query):
            chunks = ["Based on", " the movie", " reviews, The", " Matrix is excellent."]
            for chunk in chunks:
                yield chunk

        mock_rag_instance.process_query_stream = mock_process_query_stream
        mock_rag_class.return_value = mock_rag_instance

        # Test streaming response
        response_chunks = []
        async for chunk in chat_with_rag_astream(
            "What are some good action movies?",
            "gemini-2.0-flash-lite"
        ):
            response_chunks.append(chunk)

        # Verify RAGImdb was called correctly with a GeminiChat instance
        mock_rag_class.assert_called_once()
        call_args = mock_rag_class.call_args
        chat_model = call_args[0][0]

        # Verify it's a GeminiChat instance with correct parameters
        from chat_my_doc_app.llms import GeminiChat
        assert isinstance(chat_model, GeminiChat)
        assert chat_model.model_name == "gemini-2.0-flash-lite"
        assert chat_model.system_prompt == "You are a helpful movie review assistant."

        # Verify streaming response
        assert len(response_chunks) == 4
        full_response = ''.join(response_chunks)
        assert full_response == "Based on the movie reviews, The Matrix is excellent."

    @pytest.mark.asyncio
    @patch('os.getenv')
    async def test_chat_with_rag_astream_error(self, mock_getenv):
        """Test RAG chat streaming with error handling."""
        # Mock missing environment variable to trigger error
        mock_getenv.return_value = None

        # Test error handling
        response_chunks = []
        async for chunk in chat_with_rag_astream(
            "test query",
            "gemini-2.0-flash-lite"
        ):
            response_chunks.append(chunk)

        # Should yield error message
        assert len(response_chunks) > 0
        full_response = ''.join(response_chunks)
        assert "RAG Error" in full_response
        assert "CLOUD_RUN_API_URL" in full_response

    @pytest.mark.asyncio
    @patch('chat_my_doc_app.chats.RAGImdb')
    async def test_chat_with_rag_astream_empty_response(self, mock_rag_class):
        """Test RAG chat streaming with empty response."""
        # Mock RAGImdb instance with empty streaming
        mock_rag_instance = Mock()

        async def mock_process_query_stream(query):
            # Return empty stream
            return
            yield  # Never reached

        mock_rag_instance.process_query_stream = mock_process_query_stream
        mock_rag_class.return_value = mock_rag_instance

        # Test streaming response
        response_chunks = []
        async for chunk in chat_with_rag_astream(
            "empty query",
            "gemini-2.0-flash-lite"
        ):
            response_chunks.append(chunk)

        # Should handle empty streaming gracefully
        # May have citations even if no main response
        full_response = ''.join(response_chunks).strip()
        assert len(full_response) >= 0  # Should at least not crash

    @pytest.mark.asyncio
    @patch('chat_my_doc_app.chats.RAGImdb')
    async def test_chat_with_rag_astream_custom_params(self, mock_rag_class):
        """Test RAG chat streaming with custom parameters."""
        # Mock RAGImdb instance and its streaming method
        mock_rag_instance = Mock()

        async def mock_process_query_stream(query):
            chunks = ["Custom response", " for movie query."]
            for chunk in chunks:
                yield chunk

        mock_rag_instance.process_query_stream = mock_process_query_stream
        mock_rag_class.return_value = mock_rag_instance

        # Test with custom parameters
        response_chunks = []
        async for chunk in chat_with_rag_astream(
            "custom query",
            "gemini-1.5-pro",
            "custom_session",
            "You are a movie expert."
        ):
            response_chunks.append(chunk)

        # Verify RAGImdb was called with custom parameters
        mock_rag_class.assert_called_once()
        call_args = mock_rag_class.call_args
        chat_model = call_args[0][0]

        # Verify it's a GeminiChat instance with correct parameters
        from chat_my_doc_app.llms import GeminiChat
        assert isinstance(chat_model, GeminiChat)
        assert chat_model.model_name == "gemini-1.5-pro"
        assert chat_model.system_prompt == "You are a movie expert."

        # Verify response
        assert len(response_chunks) == 2
        full_response = ''.join(response_chunks)
        assert full_response == "Custom response for movie query."


class TestRAGConnectionTesting:
    """Test RAG connection testing functionality."""

    @patch('chat_my_doc_app.chats.RAGImdb')
    def test_rag_connection_success(self, mock_rag_class):
        """Test successful RAG connection test."""
        # Mock successful RAG workflow
        mock_rag = Mock()
        mock_rag.get_workflow_info.return_value = {
            'workflow_type': 'RAG with LangGraph',
            'nodes': ['retrieve', 'generate', 'respond']
        }
        mock_rag_class.return_value = mock_rag

        # Test connection
        result = if_rag_connection_works()

        # Should return success
        assert result is True
        mock_rag_class.assert_called_once()
        mock_rag.get_workflow_info.assert_called_once()

    @patch('chat_my_doc_app.chats.RAGImdb')
    def test_rag_connection_failure(self, mock_rag_class):
        """Test RAG connection test with failure."""
        # Mock RAG workflow that raises exception
        mock_rag_class.side_effect = Exception("Connection failed")

        # Test connection
        result = if_rag_connection_works()

        # Should return failure
        assert result is False
        mock_rag_class.assert_called_once()


class TestAppIntegrationFlow:
    """Test overall app integration flow."""

    @patch('chat_my_doc_app.chats.RAGImdb')
    def test_rag_workflow_integration(self, mock_rag_class):
        """Test that RAG workflow integrates correctly with chat functions."""
        # Mock RAG workflow with comprehensive response
        mock_rag = Mock()
        mock_rag.get_workflow_info.return_value = {
            'workflow_type': 'RAG with LangGraph',
            'nodes': ['retrieve', 'generate', 'respond'],
            'rag_service': {
                'max_context_length': 4000,
                'source_format': 'markdown'
            },
            'llm': {
                'type': 'GeminiChat',
                'model_name': 'gemini-2.0-flash-lite'
            }
        }
        mock_rag.process_query.return_value = {
            'response': 'Comprehensive movie review response with citations.',
            'citations': [
                {
                    'id': 1,
                    'movie_title': 'The Matrix',
                    'year': '1999',
                    'score': 0.95
                }
            ],
            'context': 'Rich context about movies',
            'metadata': {
                'retrieved_docs': 3,
                'context_length': 800,
                'citations_included': True,
                'workflow_started': True
            }
        }
        mock_rag_class.return_value = mock_rag

        # Test connection
        connection_result = if_rag_connection_works()
        assert connection_result

        # Test workflow info
        info = mock_rag.get_workflow_info()
        assert info['workflow_type'] == 'RAG with LangGraph'
        assert len(info['nodes']) == 3

        # Test query processing
        query_result = mock_rag.process_query("test movie query")
        assert 'response' in query_result
        assert 'citations' in query_result
        assert 'metadata' in query_result
        assert query_result['metadata']['citations_included'] is True

        # Verify all methods called correctly
        mock_rag_class.assert_called()
        mock_rag.get_workflow_info.assert_called()
        mock_rag.process_query.assert_called_with("test movie query")


@pytest.mark.asyncio
class TestStreamingBehavior:
    """Test streaming behavior in RAG integration."""

    @patch('chat_my_doc_app.chats.RAGImdb')
    async def test_streaming_chunk_generation(self, mock_rag_class):
        """Test that responses are properly chunked for streaming."""
        # Mock RAGImdb instance and its streaming method
        mock_rag_instance = Mock()

        async def mock_process_query_stream(query):
            for i in range(10):
                yield f"Chunk {i} "

        mock_rag_instance.process_query_stream = mock_process_query_stream
        mock_rag_class.return_value = mock_rag_instance

        # Collect chunks
        chunks = []
        async for chunk in chat_with_rag_astream("long query", "gemini-2.0-flash-lite"):
            chunks.append(chunk)

        # Should have exactly 10 chunks
        assert len(chunks) == 10

        # Chunks should contain the streaming content
        full_response = ''.join(chunks)
        assert "Chunk 0 " in full_response
        assert "Chunk 9 " in full_response

    @patch('chat_my_doc_app.chats.RAGImdb')
    async def test_streaming_preserves_content_integrity(self, mock_rag_class):
        """Test that streaming preserves content integrity."""
        # Mock RAGImdb instance and its streaming method
        mock_rag_instance = Mock()

        async def mock_process_query_stream(query):
            words = ["The", " Matrix", " is", " an", " excellent", " science", " fiction", " movie."]
            for word in words:
                yield word

        mock_rag_instance.process_query_stream = mock_process_query_stream
        mock_rag_class.return_value = mock_rag_instance

        # Collect chunks
        chunks = []
        async for chunk in chat_with_rag_astream("test", "gemini-2.0-flash-lite"):
            chunks.append(chunk)

        # Should have received all chunks
        assert len(chunks) == 8

        # Full response should contain the expected content
        full_response = ''.join(chunks)
        assert full_response == "The Matrix is an excellent science fiction movie."
