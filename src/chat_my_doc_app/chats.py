"""
Chat functionality using custom LangChain implementation for deployed Gemini API.

This module provides chat functionality with conversation memory using
a custom LangChain BaseChatModel that connects to your deployed API.
"""
import os
from typing import Annotated, AsyncIterator, Iterator, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, add_messages, START, END
from loguru import logger

from chat_my_doc_app.llms import GeminiChat

load_dotenv()

# State definition for LangGraph
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Initialize LangGraph with InMemorySaver for short-term memory
checkpointer = InMemorySaver()

async def chat_node(state: State, config) -> State:
    """Chat node that processes messages and generates responses with streaming."""
    # Get API URL from environment variable
    api_url = os.getenv("CLOUD_RUN_API_URL")
    if not api_url:
        raise ValueError("CLOUD_RUN_API_URL environment variable is not set")
    
    # Get model name from config metadata, fallback to default
    model_name = config.get("configurable", {}).get("model_name", "gemini-2.0-flash-lite")
    logger.debug(f"Using model: {model_name}")
    
    # Configure the custom LangChain model
    llm = GeminiChat(
        api_url=api_url,    
        model_name=model_name
    )

    # Generate response using all messages as context (async)
    response = await llm.ainvoke(state["messages"], config)
    
    # Return updated state with the new response
    return {"messages": [response]}

# Compile the graph with checkpointer
graph = (
    StateGraph(State)
    .add_node("chat", chat_node)
    .add_edge(START, "chat")
    .add_edge("chat", END)
).compile(checkpointer=checkpointer)

async def chat_with_gemini_astream(
    message: str,
    model_name: str,
    session_id: str = "default"
) -> AsyncIterator[str]:
    """
    Chat with deployed Gemini API using LangGraph native streaming.
    
    Args:
        message: User message
        model_name: Gemini model to use
        session_id: Session identifier for conversation history
        
    Yields:
        str: Streaming response chunks
    """
    try:
        # Create user message
        user_message = HumanMessage(content=message)
        
        # Configure thread and model
        config = {
            "configurable": {
                "thread_id": session_id,
                "model_name": model_name
            }
        }
        
        logger.debug(f"Session ID: {session_id}")
        logger.debug(f"Model Name: {model_name}")
        
        # Stream tokens using LangGraph's native async streaming with messages mode
        async for message_chunk, _ in graph.astream(
            {"messages": [user_message]},
            config,
            stream_mode="messages"
        ):
            if hasattr(message_chunk, 'content') and message_chunk.content:
                content = message_chunk.content
                # Handle different content types
                if isinstance(content, str):
                    logger.debug(f"Streaming chunk which is a string: {content}")
                    yield content
                else:
                    raise ValueError(f"Unexpected content type: {type(content)}")
            else:
                raise ValueError("Message chunk does not have content")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        yield error_msg

def get_conversation_history(session_id: str) -> List[BaseMessage]:
    """Get conversation history for a session using LangGraph state."""
    config = {"configurable": {"thread_id": session_id}}
    current_state = graph.get_state(config)
    return current_state.values.get("messages", []) if current_state.values else []

def clear_conversation_history(session_id: str) -> None:
    """Clear conversation history for a session."""
    config = {"configurable": {"thread_id": session_id}}
    # Update state with empty messages list
    graph.update_state(config, {"messages": []})
    
    logger.debug(f"Conversation history cleared for session: {session_id}")
    logger.debug(f"Current state after clearing: {graph.get_state(config)}")

def get_available_models() -> List[str]:
    """Get list of available Gemini models."""
    return [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash", 
        "gemini-1.5-pro"
    ]