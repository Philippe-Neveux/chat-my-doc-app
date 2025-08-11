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
import typer
from dotenv import load_dotenv
from loguru import logger
from typing_extensions import Annotated

from chat_my_doc_app.chats import (
    chat_with_gemini_astream,
    clear_conversation_history,
    get_available_models,
)

load_dotenv()

def create_chat_interface():
    """Create the Gradio chat interface."""
    
    # Use a single session ID for the entire conversation
    session_id = "gradio_session_1"
    
    async def respond(
        message: str,
        history: List[dict],
        model_name: str,
        system_prompt: str
    ):
        """Handle user message and generate response."""
        if not message.strip():
            yield "", history
            return 
        
        # Stream the response
        partial_response = ""
        async for chunk in chat_with_gemini_astream(
            message, model_name, session_id, system_prompt
        ):
            partial_response += chunk
            # Update the history with current partial response in messages format
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial_response}
            ]
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
                    type="messages"
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
                
                system_prompt_input = gr.Textbox(
                    value="You are a helpful assistant.",
                    label="System Prompt",
                    placeholder="Enter system prompt...",
                    lines=3,
                    interactive=True
                )
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Event handlers
        msg.submit(
            respond,
            inputs=[msg, chatbot, model_dropdown, system_prompt_input],
            outputs=[msg, chatbot]
        )
        
        send_btn.click(
            respond,
            inputs=[msg, chatbot, model_dropdown, system_prompt_input], 
            outputs=[msg, chatbot]
        )
        
        def clear_history():
            # Clear the conversation history for this session
            clear_conversation_history(session_id)
            logger.debug("Conversation history cleared")
            return [], ""
        
        clear_btn.click(
            clear_history,
            outputs=[chatbot, msg]
        )
    
    return interface

# Create Typer app
app = typer.Typer(help="Chat My Doc App - Gradio interface for chatting with Gemini AI")

def validate_port(value: int | None) -> int | None:
    """Validate port number is in valid range."""
    if value is None:
        return None
    if not 1 <= value <= 65535:
        raise typer.BadParameter("Port must be between 1 and 65535")
    return value

def validate_host(value: str) -> str:
    """Validate host format."""
    import socket
    if value not in ["0.0.0.0", "localhost", "127.0.0.1"]:
        # Try to validate as IP address
        try:
            socket.inet_aton(value)
        except socket.error:
            raise typer.BadParameter(f"Invalid host format: {value}")
    return value

@app.command()
def main(
    debug: bool = typer.Option(
        False, 
        "--debug", 
        "-d", 
        help="Enable debug mode with auto-reload and detailed error messages"
    ),
    port: Annotated[int | None, typer.Option(
        "--port",
        "-p",
        help="Port to run the server on (overrides PORT environment variable)",
        callback=validate_port,
        min=1,
        max=65535
    )] = None,
    host: Annotated[str, typer.Option(
        "--host",
        help="Host to bind the server to",
        callback=validate_host
    )] = "0.0.0.0",
    share: bool = typer.Option(
        False, 
        "--share", 
        "-s", 
        help="Create a public shareable link"
    ),
    browser: bool = typer.Option(
        False, 
        "--browser", 
        "-b", 
        help="Auto-open in browser (useful for development)"
    ),
):
    """Launch the Chat My Doc App Gradio interface."""
    typer.echo("🚀 Starting Chat My Doc App...")
    
    interface = create_chat_interface()
    
    # Determine port: CLI argument > environment variable > default
    if port is None:
        port = int(os.getenv("PORT", 8000))
    
    typer.echo(f"🌐 Server will run on {host}:{port}")
    if debug:
        typer.echo("🐛 Debug mode enabled - auto-reload and detailed errors")
    if share:
        typer.echo("🔗 Creating public shareable link")
    
    # Launch the interface
    interface.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        debug=debug,
        inbrowser=browser
    )

if __name__ == "__main__":
    app()