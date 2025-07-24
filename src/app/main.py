"""
Chat My Doc App - Main Application Module

This module provides a Chainlit-based chat interface that connects to a custom
Google Cloud Run LLM API with conversation memory capabilities.

The application maintains conversation history per user session and sends the
complete context to the LLM for coherent, context-aware responses.
"""

import os

import chainlit as cl
from dotenv import load_dotenv
from loguru import logger

from app.memory import (
    get_conversation_history_with_current_prompt,
    update_conversation_history_from_llm_content,
)
from chat_my_doc_app.llms import CloudRunLLM

load_dotenv()

CLOUD_RUN_API_URL = os.getenv("CLOUD_RUN_API_URL")

if not CLOUD_RUN_API_URL:
    raise ValueError("CLOUD_RUN_API_URL environment variable is not set")

llm = CloudRunLLM(api_url=CLOUD_RUN_API_URL)

@cl.on_chat_start
async def start():
    """
    Initialize a new chat session.
    
    This function is called when a user starts a new chat session. It initializes
    an empty conversation history in the user session and sends a welcome message.
    
    The conversation history is stored per user session to maintain context
    throughout the conversation.
    """
    # Initialize conversation history in user session
    cl.user_session.set("conversation_history", [])
    
    await cl.Message(
        content="Hello! I'm your AI assistant powered by Gemini. How can I help you today?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming user messages with conversation memory.
    
    This function processes each user message by:
    1. Retrieving and updating the conversation history
    2. Building a complete conversation context
    3. Sending the context to the LLM via Google Cloud Run API
    4. Storing the LLM response in conversation history
    5. Sending the response back to the user
    
    Args:
        message (cl.Message): The incoming user message from Chainlit
        
    Raises:
        Exception: Any errors during LLM invocation or message processing
        
    Note:
        The conversation history is maintained per user session and includes
        both user messages and assistant responses for context continuity.
    """
    if not llm:
        await cl.Message(
            content="Error: CLOUD_RUN_API_URL environment variable not set"
        ).send()
        return
        
    try:
        # Get conversation history with current prompt included
        prompt = get_conversation_history_with_current_prompt(message.content)
        
        # Invoke LLM with full conversation context
        response = llm.invoke(prompt)
        llm_content = str(response)
        
        # Update conversation history with LLM response
        update_conversation_history_from_llm_content(llm_content)
        
        # Log conversation state for debugging
        logger.debug(f"Conversation history: {cl.user_session.get('conversation_history')}")
        
        # Send response to user
        await cl.Message(
            content=llm_content
        ).send()
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await cl.Message(
            content=f"Sorry, I encountered an error: {str(e)}"
        ).send()
