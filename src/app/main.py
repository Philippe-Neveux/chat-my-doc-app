"""
Chat My Doc App - Main Application Module

This module provides a Gradio-based chat interface that connects directly
to Google's Gemini API with conversation memory capabilities.

The application maintains conversation history per user session and sends the
complete context to the LLM for coherent, context-aware responses.
"""

import os
from typing import List, Dict, Any, Iterator

import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv
from loguru import logger

from app.memory import (
    get_conversation_history, 
    add_to_history, 
    build_conversation_context
)

load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

client = genai.Client()

# Available models
AVAILABLE_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash", 
    "gemini-1.5-pro"
]

# Global conversation history storage (in production, use proper session management)
conversation_histories: Dict[str, List[types.Content]] = {}

def chat_with_gemini(
    message: str,
    model_name: str,
    session_id: str = "default"
) -> Iterator[str]:
    """Chat with Gemini API with streaming support."""
    try:
        # Add user message to history
        add_to_history(session_id, "user", message, conversation_histories)
        
        # Get conversation history
        context_history = get_conversation_history(session_id, conversation_histories)
            
        logger.debug(f"Using model: {model_name}")
        logger.debug(f"Full prompt: {context_history}")
        
        # Generate streaming response
        response = client.models.generate_content_stream(
            model=model_name,
            contents=context_history,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text
        
        # Add assistant response to history
        add_to_history(session_id, "model", full_response, conversation_histories)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        yield error_msg

def create_chat_interface():
    """Create the Gradio chat interface."""
    
    # Use a single session ID for the entire conversation
    session_id = "gradio_session_1"
    
    def respond(
        message: str,
        history: List[List[str]],
        model_name: str
    ):
        """Handle user message and generate response."""
        if not message.strip():
            return "", history
        
        # Stream the response
        partial_response = ""
        for chunk in chat_with_gemini(message, model_name, session_id):
            partial_response += chunk
            # Update the history with current partial response
            new_history = history + [[message, partial_response]]
            yield "", new_history
    
    # Create the interface
    with gr.Blocks(title="Chat My Doc App") as interface:
        gr.Markdown("# Chat My Doc App")
        gr.Markdown("Chat with Google's Gemini AI models")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    height=600,
                    show_label=False,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=4,
                        container=False
                    )
                    send_btn = gr.Button("Send", scale=1)
                    
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[0],
                    label="Select Model",
                    interactive=True
                )
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Event handlers
        msg.submit(
            respond,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[msg, chatbot]
        )
        
        send_btn.click(
            respond,
            inputs=[msg, chatbot, model_dropdown], 
            outputs=[msg, chatbot]
        )
        
        def clear_history():
            # Clear the conversation history for this session
            if session_id in conversation_histories:
                conversation_histories[session_id] = []
            return [], ""
        
        clear_btn.click(
            clear_history,
            outputs=[chatbot, msg]
        )
    
    return interface

def main():
    """Main entry point for the application."""
    interface = create_chat_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()