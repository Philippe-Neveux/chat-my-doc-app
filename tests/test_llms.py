import pytest
from unittest.mock import Mock, patch
import json

from chat_my_doc_app.llms import CloudRunLLM


class TestCloudRunLLM:
    """Test suite for CloudRunLLM class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_url = "https://test-api.example.com"
        self.llm = CloudRunLLM(api_url=self.api_url)
    
    def test_initialization(self):
        """Test CloudRunLLM initialization."""
        assert self.llm.api_url == self.api_url
    
    def test_llm_type_property(self):
        """Test the _llm_type property returns correct value."""
        assert self.llm._llm_type == "cloud_run_llm"
    
    @patch('requests.post')
    def test_call_success_with_message_content(self, mock_post):
        """Test successful API call with message.content response format."""
        # Mock successful response with nested content
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "Hello, this is a test response"
            }
        }
        mock_post.return_value = mock_response
        
        result = self.llm._call("test prompt", model_name="a model name")
        
        assert result == "Hello, this is a test response"
        mock_post.assert_called_once_with(
            f"{self.api_url}/gemini-model",
            params={"prompt": "test prompt", "model_name": "a model name"}
        )
    
    @patch('requests.post')
    def test_call_success_with_string_response(self, mock_post):
        """Test successful API call with direct string response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = "Direct string response"
        mock_post.return_value = mock_response
        
        result = self.llm._call("test prompt")
        
        assert result == "Direct string response"
    
    @patch('requests.post')
    def test_call_success_with_dict_fallback(self, mock_post):
        """Test successful API call with dict response (no known fields)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "unknown_field": "some value",
            "another_field": "another value"
        }
        mock_post.return_value = mock_response
        
        result = self.llm._call("test prompt")
        
        # Should return string representation of the dict
        assert "unknown_field" in result
        assert "some value" in result
    
    @patch('requests.post')
    def test_call_api_error_status(self, mock_post):
        """Test API call with error status code."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        
        result = self.llm._call("test prompt")
        
        assert result == "API Error: 400"
    
    @patch('requests.post')
    def test_call_api_error_500(self, mock_post):
        """Test API call with server error status code."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        result = self.llm._call("test prompt")
        
        assert result == "API Error: 500"
    
    @patch('requests.post')
    def test_call_request_exception(self, mock_post):
        """Test API call with request exception."""
        mock_post.side_effect = Exception("Connection failed")
        
        result = self.llm._call("test prompt")
        
        assert result == "Error: Connection failed"
    
    @patch('requests.post')
    def test_call_json_decode_error(self, mock_post):
        """Test API call with JSON decode error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        result = self.llm._call("test prompt")
        
        assert "Error:" in result
        assert "Invalid JSON" in result
    
    @patch('requests.post')
    def test_call_with_stop_parameter(self, mock_post):
        """Test API call with stop parameter (should be ignored)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "Response with stop parameter"
            }
        }
        mock_post.return_value = mock_response
        
        result = self.llm._call(
            "test prompt",
            model_name='a model',
            stop=["stop1", "stop2"]
        )
        
        assert result == "Response with stop parameter"
        # Verify the stop parameter doesn't affect the API call
        mock_post.assert_called_once_with(
            f"{self.api_url}/gemini-model",
            params={"prompt": "test prompt", "model_name": "a model"}
        )
    
    @patch('requests.post')
    def test_call_with_kwargs(self, mock_post):
        """Test API call with additional kwargs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "Response with kwargs"
            }
        }
        mock_post.return_value = mock_response
        
        result = self.llm._call(
            "test prompt", 
            temperature=0.5, 
            max_tokens=100
        )
        
        assert result == "Response with kwargs"
    
    def test_initialization_with_empty_api_url(self):
        """Test initialization with empty API URL."""
        llm = CloudRunLLM(api_url="")
        assert llm.api_url == ""
    
    @patch('requests.post')
    def test_call_empty_prompt(self, mock_post):
        """Test API call with empty prompt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "Empty prompt response"
            }
        }
        mock_post.return_value = mock_response
        
        result = self.llm._call("", model_name="a model name")
        
        assert result == "Empty prompt response"
        mock_post.assert_called_once_with(
            f"{self.api_url}/gemini-model",
            params={"prompt": "", "model_name": "a model name"}
        )
    
    @patch('requests.post')
    def test_call_with_none_response_content(self, mock_post):
        """Test API call when message content is None."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": None
            }
        }
        mock_post.return_value = mock_response
        
        result = self.llm._call("test prompt")
        
        # Should handle None content gracefully
        assert result == "None"


class TestCloudRunLLMIntegration:
    """Integration tests for CloudRunLLM."""
    
    def test_langchain_compatibility(self):
        """Test that CloudRunLLM is compatible with LangChain interfaces."""
        llm = CloudRunLLM(api_url="https://test-api.example.com")
        
        # Test that it has required LangChain LLM attributes/methods
        assert hasattr(llm, '_call')
        assert hasattr(llm, '_llm_type')
        assert hasattr(llm, 'invoke')
        assert callable(llm._call)
        assert isinstance(llm._llm_type, str)