"""
Chat My Doc App - Main Application Module

This module provides a Gradio-based chat interface that uses LangChain
to connect to Google's Gemini API with conversation memory capabilities.

The application maintains conversation history per user session and sends the
complete context to the LLM for coherent, context-aware responses.
"""

import os
from typing import List

import gradio as gr
from dotenv import load_dotenv

from chat_my_doc_app.chats import (
    chat_with_gemini_stream,
    clear_conversation_history,
    get_available_models,
)

load_dotenv()

# Configure Google API Key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable is not set")

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
        for chunk in chat_with_gemini_stream(message, model_name, session_id):
            partial_response += chunk
            # Update the history with current partial response
            new_history = history + [[message, partial_response]]
            yield "", new_history
    
    # Create the interface
    with gr.Blocks(title="Chat My Doc App") as interface:
        gr.Markdown("# Chat My Doc App")
        gr.Markdown("Chat with Google's Gemini AI models using LangChain")
        
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
                    choices=get_available_models(),
                    value=get_available_models()[0],
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
            clear_conversation_history(session_id)
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