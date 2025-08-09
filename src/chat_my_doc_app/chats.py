"""
Chat functionality using custom LangChain implementation for deployed Gemini API.

This module provides chat functionality with conversation memory using
a custom LangChain BaseChatModel that connects to your deployed API.
"""

import os
from typing import List, Dict, Iterator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
from loguru import logger

from .custom_llm import CustomGeminiChat

load_dotenv()

# Available models
AVAILABLE_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash", 
    "gemini-1.5-pro"
]

# Global conversation history storage (in production, use proper session management)
conversation_histories: Dict[str, List[BaseMessage]] = {}

def get_conversation_history(session_id: str) -> List[BaseMessage]:
    """Get conversation history for a session."""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return conversation_histories[session_id]

def add_to_history(session_id: str, message: BaseMessage) -> None:
    """Add a message to conversation history."""
    history = get_conversation_history(session_id)
    history.append(message)

def clear_conversation_history(session_id: str) -> None:
    """Clear conversation history for a session."""
    if session_id in conversation_histories:
        conversation_histories[session_id] = []

def chat_with_gemini_stream(
    message: str,
    model_name: str,
    session_id: str = "default"
) -> Iterator[str]:
    """
    Chat with deployed Gemini API using custom LangChain implementation with streaming support.
    
    Args:
        message: User message
        model_name: Gemini model to use
        session_id: Session identifier for conversation history
        
    Yields:
        str: Streaming response chunks
    """
    try:
        # Get API URL from environment variable
        api_url = os.getenv("CLOUD_RUN_API_URL")
        if not api_url:
            raise ValueError("CLOUD_RUN_API_URL environment variable is not set")
        
        # Configure the custom LangChain model
        llm = CustomGeminiChat(
            api_url=api_url,
            model_name=model_name
        )
        
        # Add user message to history
        user_message = HumanMessage(content=message)
        add_to_history(session_id, user_message)
        
        # Get conversation history
        history = get_conversation_history(session_id)
        
        logger.debug(f"Using model: {model_name}")
        logger.debug(f"Conversation history length: {len(history)}")
        
        # Stream the response
        full_response = ""
        for chunk in llm.stream(history):
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                # Handle different content types
                if isinstance(content, str):
                    full_response += content
                    yield content
                elif isinstance(content, list):
                    # Convert list to string
                    text_content = " ".join(str(item) for item in content if item)
                    full_response += text_content
                    yield text_content
        
        # Add assistant response to history
        assistant_message = AIMessage(content=full_response)
        add_to_history(session_id, assistant_message)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        yield error_msg

def get_available_models() -> List[str]:
    """Get list of available Gemini models."""
    return AVAILABLE_MODELS.copy()