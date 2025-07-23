import os

import chainlit as cl
from dotenv import load_dotenv

from chat_my_doc_app.llms import CloudRunLLM

load_dotenv()

CLOUD_RUN_API_URL = os.getenv("CLOUD_RUN_API_URL")

if not CLOUD_RUN_API_URL:
    raise ValueError("CLOUD_RUN_API_URL environment variable is not set")

llm = CloudRunLLM(api_url=CLOUD_RUN_API_URL)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! I'm your AI assistant powered by Gemini. How can I help you today?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    if not llm:
        await cl.Message(
            content="Error: CLOUD_RUN_API_URL environment variable not set"
        ).send()
        return
        
    try:
        response = llm.invoke(message.content)
        
        # Extract content from LangChain response object
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        await cl.Message(
            content=content
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Sorry, I encountered an error: {str(e)}"
        ).send()
