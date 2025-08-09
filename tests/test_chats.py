"""Tests for chats module functionality."""
import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage, AIMessage

from chat_my_doc_app.chats import (
    get_conversation_history,
    add_to_history,
    clear_conversation_history,
    chat_with_gemini_stream,
    get_available_models
)


class TestConversationHistory:
    """Test conversation history management functions."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing conversation histories
        from chat_my_doc_app.chats import conversation_histories
        conversation_histories.clear()
    
    def test_get_conversation_history_new_session(self):
        """Test getting conversation history for a new session."""
        session_id = "test_session_1"
        history = get_conversation_history(session_id)
        
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_get_conversation_history_existing_session(self):
        """Test getting conversation history for an existing session."""
        session_id = "test_session_2"
        
        # First call creates the session
        history1 = get_conversation_history(session_id)
        
        # Add a message to history
        test_message = HumanMessage(content="Hello")
        add_to_history(session_id, test_message)
        
        # Second call should return the same history with the message
        history2 = get_conversation_history(session_id)
        
        assert history1 is history2
        assert len(history2) == 1
        assert history2[0].content == "Hello"
    
    def test_add_to_history_human_message(self):
        """Test adding a human message to conversation history."""
        session_id = "test_session_3"
        message = HumanMessage(content="How are you?")
        
        add_to_history(session_id, message)
        history = get_conversation_history(session_id)
        
        assert len(history) == 1
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "How are you?"
    
    def test_add_to_history_ai_message(self):
        """Test adding an AI message to conversation history."""
        session_id = "test_session_4"
        message = AIMessage(content="I'm doing well, thank you!")
        
        add_to_history(session_id, message)
        history = get_conversation_history(session_id)
        
        assert len(history) == 1
        assert isinstance(history[0], AIMessage)
        assert history[0].content == "I'm doing well, thank you!"
    
    def test_add_to_history_multiple_messages(self):
        """Test adding multiple messages to conversation history."""
        session_id = "test_session_5"
        
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")
        human_msg2 = HumanMessage(content="How are you?")
        
        add_to_history(session_id, human_msg)
        add_to_history(session_id, ai_msg)
        add_to_history(session_id, human_msg2)
        
        history = get_conversation_history(session_id)
        
        assert len(history) == 3
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"
        assert history[2].content == "How are you?"
    
    def test_clear_conversation_history_existing_session(self):
        """Test clearing conversation history for an existing session."""
        session_id = "test_session_6"
        
        # Add some messages
        add_to_history(session_id, HumanMessage(content="Hello"))
        add_to_history(session_id, AIMessage(content="Hi!"))
        
        # Verify messages exist
        history = get_conversation_history(session_id)
        assert len(history) == 2
        
        # Clear history
        clear_conversation_history(session_id)
        
        # Verify history is cleared
        history_after_clear = get_conversation_history(session_id)
        assert len(history_after_clear) == 0
    
    def test_clear_conversation_history_nonexistent_session(self):
        """Test clearing conversation history for a nonexistent session."""
        session_id = "nonexistent_session"
        
        # This should not raise an error
        clear_conversation_history(session_id)
        
        # Should still be able to get history (will create empty list)
        history = get_conversation_history(session_id)
        assert len(history) == 0
    
    def test_separate_session_histories(self):
        """Test that different sessions maintain separate histories."""
        session_1 = "session_1"
        session_2 = "session_2"
        
        add_to_history(session_1, HumanMessage(content="Session 1 message"))
        add_to_history(session_2, HumanMessage(content="Session 2 message"))
        
        history_1 = get_conversation_history(session_1)
        history_2 = get_conversation_history(session_2)
        
        assert len(history_1) == 1
        assert len(history_2) == 1
        assert history_1[0].content == "Session 1 message"
        assert history_2[0].content == "Session 2 message"
        
        # Clear one session should not affect the other
        clear_conversation_history(session_1)
        
        history_1_after = get_conversation_history(session_1)
        history_2_after = get_conversation_history(session_2)
        
        assert len(history_1_after) == 0
        assert len(history_2_after) == 1
        assert history_2_after[0].content == "Session 2 message"


class TestChatFunctions:
    """Test chat-related functions."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing conversation histories
        from chat_my_doc_app.chats import conversation_histories
        conversation_histories.clear()
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemini-2.0-flash-lite" in models
        assert "gemini-2.0-flash" in models
        assert "gemini-1.5-pro" in models
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.CustomGeminiChat')
    def test_chat_with_gemini_stream_success(self, mock_chat_class):
        """Test successful chat with Gemini stream."""
        # Mock the CustomGeminiChat instance and its stream method
        mock_llm = Mock()
        mock_chat_class.return_value = mock_llm
        mock_llm.stream.return_value = [
            Mock(content="Hello"),
            Mock(content=" world"),
            Mock(content="!")
        ]
        
        # Test the chat function with a unique session ID
        import uuid
        session_id = f"test_session_{uuid.uuid4()}"
        result = list(chat_with_gemini_stream(
            message="Hello", 
            model_name="gemini-2.0-flash-lite",
            session_id=session_id
        ))
        
        # Verify results
        assert result == ["Hello", " world", "!"]
        
        # Verify CustomGeminiChat was initialized correctly
        mock_chat_class.assert_called_once_with(
            api_url="https://test-api.example.com",
            model_name="gemini-2.0-flash-lite"
        )
        
        # Verify stream was called with the correct history
        mock_llm.stream.assert_called_once()
        call_args = mock_llm.stream.call_args[0][0]  # Get the messages argument
        
        # Debug: Print what was actually called
        print(f"DEBUG: call_args length: {len(call_args)}")
        for i, msg in enumerate(call_args):
            print(f"DEBUG: Message {i}: {type(msg)} - {msg.content}")
            
        assert len(call_args) == 1  # Should have exactly one message (the user message)
        assert isinstance(call_args[0], HumanMessage)
        assert call_args[0].content == "Hello"
    
    @patch.dict('os.environ', {})  # Empty environment
    def test_chat_with_gemini_stream_missing_api_url(self):
        """Test chat function when API URL is missing."""
        result = list(chat_with_gemini_stream(
            message="Hello",
            model_name="gemini-2.0-flash-lite",
            session_id="test_session"
        ))
        
        assert len(result) == 1
        assert "CLOUD_RUN_API_URL environment variable is not set" in result[0]
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.CustomGeminiChat')
    def test_chat_with_gemini_stream_with_conversation_history(self, mock_chat_class):
        """Test chat function with existing conversation history."""
        session_id = "test_session_with_history"
        
        # Add some existing history
        add_to_history(session_id, HumanMessage(content="Previous question"))
        add_to_history(session_id, AIMessage(content="Previous answer"))
        
        # Mock the LLM
        mock_llm = Mock()
        mock_chat_class.return_value = mock_llm
        mock_llm.stream.return_value = [Mock(content="Response")]
        
        # Test the chat function
        result = list(chat_with_gemini_stream(
            message="New question",
            model_name="gemini-2.0-flash-lite", 
            session_id=session_id
        ))
        
        # Verify the conversation history was passed correctly
        mock_llm.stream.assert_called_once()
        call_args = mock_llm.stream.call_args[0][0]  # Get the messages argument
        
        # Should have 3 messages: previous human, previous AI, current human
        assert len(call_args) == 3
        assert call_args[0].content == "Previous question"
        assert call_args[1].content == "Previous answer" 
        assert call_args[2].content == "New question"
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.CustomGeminiChat')
    def test_chat_with_gemini_stream_exception(self, mock_chat_class):
        """Test chat function when an exception occurs."""
        # Mock CustomGeminiChat to raise an exception
        mock_chat_class.side_effect = Exception("Connection error")
        
        result = list(chat_with_gemini_stream(
            message="Hello",
            model_name="gemini-2.0-flash-lite",
            session_id="test_session"
        ))
        
        assert len(result) == 1
        assert "Error: Connection error" in result[0]
    
    @patch.dict('os.environ', {'CLOUD_RUN_API_URL': 'https://test-api.example.com'})
    @patch('chat_my_doc_app.chats.CustomGeminiChat')
    def test_chat_with_gemini_stream_updates_history(self, mock_chat_class):
        """Test that chat function updates conversation history correctly."""
        session_id = "test_history_update"
        
        # Mock the LLM
        mock_llm = Mock()
        mock_chat_class.return_value = mock_llm
        mock_llm.stream.return_value = [
            Mock(content="Hello"),
            Mock(content=" there")
        ]
        
        # Call chat function
        result = list(chat_with_gemini_stream(
            message="Hi",
            model_name="gemini-2.0-flash-lite",
            session_id=session_id
        ))
        
        # Check that history was updated
        history = get_conversation_history(session_id)
        
        # Should have 2 messages: human message + AI response
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "Hi"
        assert isinstance(history[1], AIMessage)
        assert history[1].content == "Hello there"