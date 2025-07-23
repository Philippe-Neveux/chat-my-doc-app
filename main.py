import os

from dotenv import load_dotenv
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! I'm your AI assistant powered by Gemini. How can I help you today?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        response = llm.invoke([HumanMessage(content=message.content)])
        
        await cl.Message(
            content=response.content
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Sorry, I encountered an error: {str(e)}"
        ).send()
