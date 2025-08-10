import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage, AIMessage

from chat_my_doc_app.chats import (
    get_available_models,
    chat_with_gemini_stream
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
    def test_chat_with_gemini_stream_missing_api_url(self):
        """Test chat function when API URL is missing."""
        result = list(chat_with_gemini_stream(
            message="Hello",
            model_name="gemini-2.0-flash-lite",
            session_id="test_session_missing_url"
        ))
        
        assert len(result) == 1
        assert "CLOUD_RUN_API_URL environment variable is not set" in result[0]
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.GeminiChat')
    def test_chat_with_gemini_stream_success(self, mock_chat_class):
        """Test successful chat with Gemini stream"""
        # Mock the GeminiChat instance and its stream method
        mock_llm = Mock()
        mock_chat_class.return_value = mock_llm
        mock_llm.stream.return_value = [
            Mock(content="Test"),
            Mock(content=" response")
        ]
        
        # Test the chat function with a completely unique session ID
        import uuid
        session_id = f"unique_session_{uuid.uuid4().hex}"
        result = list(chat_with_gemini_stream(
            message="Test message", 
            model_name="gemini-2.0-flash-lite",
            session_id=session_id
        ))
        
        # Verify results
        assert result == ["Test", " response"]
        
        # Verify GeminiChat was initialized correctly
        mock_chat_class.assert_called_once_with(
            api_url="https://test-api.example.com",
            model_name="gemini-2.0-flash-lite"
        )
        
        # Verify stream was called at least once
        mock_llm.stream.assert_called_once()
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.GeminiChat')
    def test_chat_with_gemini_stream_exception(self, mock_chat_class):
        """Test chat function when an exception occurs."""
        # Mock GeminiChat to raise an exception
        mock_chat_class.side_effect = Exception("Test connection error")
        
        result = list(chat_with_gemini_stream(
            message="Hello",
            model_name="gemini-2.0-flash-lite",
            session_id="test_session_exception"
        ))
        
        assert len(result) == 1
        assert "Error: Test connection error" in result[0]