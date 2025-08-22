"""Tests for custom LLM implementations."""
import json
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage

from chat_my_doc_app.llms import GeminiChat, MistralChat


class TestGeminiChat:
    """Test suite for GeminiChat class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_url = "https://test-api.example.com"
        self.llm = GeminiChat(api_url=self.api_url, model_name="gemini-2.0-flash-lite")

    def test_initialization(self):
        """Test GeminiChat initialization."""
        assert self.llm.api_url == self.api_url
        assert self.llm.model_name == "gemini-2.0-flash-lite"
        assert self.llm.system_prompt == "You are a helpful assistant."

    def test_initialization_with_custom_system_prompt(self):
        """Test GeminiChat initialization with custom system prompt."""
        custom_prompt = "You are a French translator."
        llm = GeminiChat(api_url=self.api_url, model_name="gemini-2.0-flash-lite", system_prompt=custom_prompt)
        assert llm.system_prompt == custom_prompt

    def test_llm_type_property(self):
        """Test the _llm_type property returns correct value."""
        assert self.llm._llm_type == "gemini_chat"

    def test_identifying_params(self):
        """Test the _identifying_params property."""
        params = self.llm._identifying_params
        assert params["api_url"] == self.api_url
        assert params["model_name"] == "gemini-2.0-flash-lite"

    def test_messages_to_prompt_human_messages(self):
        """Test _messages_to_prompt with HumanMessage."""
        messages = [HumanMessage(content="Hello, how are you?")]
        prompt = self.llm._messages_to_prompt(messages)
        expected = "System: You are a helpful assistant.\nHuman: Hello, how are you?"
        assert prompt == expected

    def test_messages_to_prompt_ai_messages(self):
        """Test _messages_to_prompt with AIMessage."""
        messages = [AIMessage(content="I'm doing well, thank you!")]
        prompt = self.llm._messages_to_prompt(messages)
        expected = "System: You are a helpful assistant.\nAssistant: I'm doing well, thank you!"
        assert prompt == expected

    def test_messages_to_prompt_mixed_messages(self):
        """Test _messages_to_prompt with mixed message types."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?")
        ]
        prompt = self.llm._messages_to_prompt(messages)
        expected = "System: You are a helpful assistant.\nHuman: Hello\nAssistant: Hi there!\nHuman: How are you?"
        assert prompt == expected

    def test_messages_to_prompt_with_custom_system_prompt(self):
        """Test _messages_to_prompt with custom system prompt."""
        custom_llm = GeminiChat(api_url=self.api_url, model_name="gemini-2.0-flash-lite", system_prompt="You are a translator.")
        messages = [HumanMessage(content="Hello")]
        prompt = custom_llm._messages_to_prompt(messages)
        expected = "System: You are a translator.\nHuman: Hello"
        assert prompt == expected

    @patch('requests.post')
    def test_generate_success(self, mock_post):
        """Test successful _generate call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Hello, this is a test response"
        }
        mock_post.return_value = mock_response

        messages = [HumanMessage(content="test prompt")]
        result = self.llm._generate(messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello, this is a test response"
        mock_post.assert_called_once_with(
            f"{self.api_url}/gemini",
            json={"prompt": "System: You are a helpful assistant.\nHuman: test prompt", "model_name": "gemini-2.0-flash-lite"},
            headers={"Content-Type": "application/json"}
        )

    @patch('requests.post')
    def test_generate_api_error(self, mock_post):
        """Test _generate with API error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        messages = [HumanMessage(content="test prompt")]
        result = self.llm._generate(messages)

        assert len(result.generations) == 1
        assert "API Error: 400" in result.generations[0].message.content

    @patch('requests.post')
    def test_generate_exception(self, mock_post):
        """Test _generate with request exception."""
        mock_post.side_effect = Exception("Connection failed")

        messages = [HumanMessage(content="test prompt")]
        result = self.llm._generate(messages)

        assert len(result.generations) == 1
        assert "Error calling API: Connection failed" in result.generations[0].message.content

    @patch('requests.post')
    def test_stream_success(self, mock_post):
        """Test successful _stream call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = ["Hello", " world", "!"]
        mock_post.return_value = mock_response

        messages = [HumanMessage(content="test prompt")]
        chunks = list(self.llm._stream(messages))

        assert len(chunks) == 3
        assert chunks[0].message.content == "Hello"
        assert chunks[1].message.content == " world"
        assert chunks[2].message.content == "!"
        mock_post.assert_called_once_with(
            f"{self.api_url}/gemini-stream",
            json={"prompt": "System: You are a helpful assistant.\nHuman: test prompt", "model_name": "gemini-2.0-flash-lite"},
            headers={"Content-Type": "application/json"},
            stream=True
        )

    @patch('requests.post')
    def test_stream_api_error(self, mock_post):
        """Test _stream with API error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        messages = [HumanMessage(content="test prompt")]
        chunks = list(self.llm._stream(messages))

        assert len(chunks) == 1
        assert "API Error: 500" in chunks[0].message.content

    @patch('requests.post')
    def test_stream_exception(self, mock_post):
        """Test _stream with request exception."""
        mock_post.side_effect = Exception("Network error")

        messages = [HumanMessage(content="test prompt")]
        chunks = list(self.llm._stream(messages))

        assert len(chunks) == 1
        assert "Error calling API: Network error" in chunks[0].message.content

    def test_astream_exists(self):
        """Test that _astream method exists and is callable."""
        assert hasattr(self.llm, '_astream')
        assert callable(self.llm._astream)
        # Note: Async streaming tests are complex to mock properly in this setup
        # The method exists and will work with proper aiohttp mocking in integration tests

    @pytest.mark.asyncio
    @patch.object(GeminiChat, '_generate')
    async def test_agenerate(self, mock_generate):
        """Test _agenerate method."""
        # Mock the _generate method since _agenerate falls back to it
        mock_result = Mock()
        mock_generate.return_value = mock_result

        messages = [HumanMessage(content="test prompt")]
        result = await self.llm._agenerate(messages)

        assert result == mock_result
        mock_generate.assert_called_once_with(messages, None, None)


class TestGeminiChatIntegration:
    """Integration tests for GeminiChat."""

    def test_langchain_compatibility(self):
        """Test that GeminiChat is compatible with LangChain interfaces."""
        llm = GeminiChat(api_url="https://test-api.example.com")

        # Test that it has required LangChain BaseChatModel attributes/methods
        assert hasattr(llm, '_generate')
        assert hasattr(llm, '_stream')
        assert hasattr(llm, '_astream')
        assert hasattr(llm, '_llm_type')
        assert hasattr(llm, '_identifying_params')
        assert callable(llm._generate)
        assert callable(llm._stream)
        assert isinstance(llm._llm_type, str)

    def test_model_name_override(self):
        """Test that model_name can be overridden in kwargs."""
        llm = GeminiChat(api_url="https://test-api.example.com", model_name="default-model")

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": "test response"}
            mock_post.return_value = mock_response

            messages = [HumanMessage(content="test")]
            llm._generate(messages, model_name="custom-model")

            # Check that the custom model name was used
            call_args = mock_post.call_args
            assert call_args[1]['json']['model_name'] == "custom-model"


class TestMistralChat:
    """Test suite for MistralChat class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_url = "https://test-api.example.com"
        self.llm = MistralChat(api_url=self.api_url)

    def test_initialization(self):
        """Test MistralChat initialization."""
        assert self.llm.api_url == self.api_url
        assert self.llm.system_prompt == "You are a helpful assistant."

    def test_initialization_with_custom_system_prompt(self):
        """Test MistralChat initialization with custom system prompt."""
        custom_prompt = "You are a coding assistant."
        llm = MistralChat(api_url=self.api_url, system_prompt=custom_prompt)
        assert llm.system_prompt == custom_prompt

    def test_llm_type_property(self):
        """Test the _llm_type property returns correct value."""
        assert self.llm._llm_type == "mistral_chat"

    def test_endpoint_paths(self):
        """Test the endpoint path properties."""
        assert self.llm._endpoint_path == "/mistral"
        assert self.llm._stream_endpoint_path == "/mistral-stream"

    def test_identifying_params(self):
        """Test the _identifying_params property."""
        params = self.llm._identifying_params
        assert params["api_url"] == self.api_url
        assert params["model_name"] is None  # MistralChat doesn't set a default model_name

    @patch('requests.post')
    def test_generate_success(self, mock_post):
        """Test successful _generate call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Hello, this is a Mistral response"
        }
        mock_post.return_value = mock_response

        messages = [HumanMessage(content="test prompt")]
        result = self.llm._generate(messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello, this is a Mistral response"
        mock_post.assert_called_once_with(
            f"{self.api_url}/mistral",
            json={"prompt": "System: You are a helpful assistant.\nHuman: test prompt"},
            headers={"Content-Type": "application/json"}
        )

    @patch('requests.post')
    def test_stream_success(self, mock_post):
        """Test successful _stream call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = ["Bonjour", " monde", "!"]
        mock_post.return_value = mock_response

        messages = [HumanMessage(content="test prompt")]
        chunks = list(self.llm._stream(messages))

        assert len(chunks) == 3
        assert chunks[0].message.content == "Bonjour"
        assert chunks[1].message.content == " monde"
        assert chunks[2].message.content == "!"
        mock_post.assert_called_once_with(
            f"{self.api_url}/mistral-stream",
            json={"prompt": "System: You are a helpful assistant.\nHuman: test prompt"},
            headers={"Content-Type": "application/json"},
            stream=True
        )

    def test_langchain_compatibility(self):
        """Test that MistralChat is compatible with LangChain interfaces."""
        # Test that it has required LangChain BaseChatModel attributes/methods
        assert hasattr(self.llm, '_generate')
        assert hasattr(self.llm, '_stream')
        assert hasattr(self.llm, '_astream')
        assert hasattr(self.llm, '_llm_type')
        assert hasattr(self.llm, '_identifying_params')
        assert callable(self.llm._generate)
        assert callable(self.llm._stream)
        assert isinstance(self.llm._llm_type, str)
