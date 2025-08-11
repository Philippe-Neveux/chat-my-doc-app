import pytest
import pytest_asyncio
from unittest.mock import patch, Mock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage

from chat_my_doc_app.chats import (
    get_available_models,
    chat_with_gemini_astream,
    clear_conversation_history,
    get_conversation_history
)


class TestChatFunctions:

    def test_get_available_models(self):
        """Test getting available models."""
        models = get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemini-2.0-flash-lite" in models
        assert "gemini-2.0-flash" in models
        assert "gemini-1.5-pro" in models
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': ''})  # Empty API URL
    @pytest.mark.asyncio
    async def test_chat_with_gemini_astream_missing_api_url(self):
        """Test async chat function when API URL is missing."""
        result = []
        async for chunk in chat_with_gemini_astream(
            message="Hello",
            model_name="gemini-2.0-flash-lite",
            session_id="test_session_missing_url"
        ):
            result.append(chunk)
        
        assert len(result) == 1
        assert "CLOUD_RUN_API_URL environment variable is not set" in result[0]
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.graph')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_astream_success(self, mock_graph):
        """Test successful async chat with Gemini stream using LangGraph"""
        # Mock the graph.astream method to return async generator
        async def mock_astream(*args, **kwargs):
            # Simulate streaming response chunks
            yield (Mock(content="Test"), {})
            yield (Mock(content=" response"), {})
        
        mock_graph.astream = mock_astream
        
        # Test the chat function with a completely unique session ID
        import uuid
        session_id = f"unique_session_{uuid.uuid4().hex}"
        result = []
        async for chunk in chat_with_gemini_astream(
            message="Test message", 
            model_name="gemini-2.0-flash-lite",
            session_id=session_id
        ):
            result.append(chunk)
        
        # Verify results
        assert result == ["Test", " response"]
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.graph')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_astream_exception(self, mock_graph):
        """Test async chat function when an exception occurs."""
        # Mock graph.astream to raise an exception
        async def mock_astream_error(*args, **kwargs):
            raise Exception("Test connection error")
            yield  # unreachable but needed for generator
        
        mock_graph.astream = mock_astream_error
        
        result = []
        async for chunk in chat_with_gemini_astream(
            message="Hello",
            model_name="gemini-2.0-flash-lite",
            session_id="test_session_exception"
        ):
            result.append(chunk)
        
        assert len(result) == 1
        assert "Error: Test connection error" in result[0]
    
    @patch('chat_my_doc_app.chats.graph')
    def test_clear_conversation_history(self, mock_graph):
        """Test clearing conversation history."""
        mock_graph.update_state = Mock()
        
        session_id = "test_session"
        clear_conversation_history(session_id)
        
        # Verify update_state was called with correct parameters
        expected_config = {"configurable": {"thread_id": session_id}}
        expected_state = {"messages": []}
        mock_graph.update_state.assert_called_once_with(expected_config, expected_state)
    
    @patch('chat_my_doc_app.chats.graph')
    def test_get_conversation_history_empty(self, mock_graph):
        """Test getting conversation history when empty."""
        mock_state = Mock()
        mock_state.values = {"messages": []}
        mock_graph.get_state.return_value = mock_state
        
        session_id = "test_session"
        history = get_conversation_history(session_id)
        
        assert history == []
        
        # Verify get_state was called with correct config
        expected_config = {"configurable": {"thread_id": session_id}}
        mock_graph.get_state.assert_called_once_with(expected_config)
    
    @patch('chat_my_doc_app.chats.graph')
    def test_get_conversation_history_with_messages(self, mock_graph):
        """Test getting conversation history with existing messages."""
        # Create mock messages
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")
        
        mock_state = Mock()
        mock_state.values = {"messages": [human_msg, ai_msg]}
        mock_graph.get_state.return_value = mock_state
        
        session_id = "test_session"
        history = get_conversation_history(session_id)
        
        assert len(history) == 2
        assert history[0] == human_msg
        assert history[1] == ai_msg
        
        # Verify get_state was called with correct config
        expected_config = {"configurable": {"thread_id": session_id}}
        mock_graph.get_state.assert_called_once_with(expected_config)
    
    @patch('chat_my_doc_app.chats.graph')
    def test_get_conversation_history_no_state(self, mock_graph):
        """Test getting conversation history when no state exists."""
        mock_state = Mock()
        mock_state.values = None
        mock_graph.get_state.return_value = mock_state
        
        session_id = "test_session"
        history = get_conversation_history(session_id)
        
        assert history == []