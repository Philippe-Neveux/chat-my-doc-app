"""
Custom LangChain Chat Models for deployed Gateway APIs.

This module provides custom LangChain BaseChatModel implementations
that connect to your deployed Gateway API with streaming support for different LLM providers.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, List, Optional

import aiohttp
import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from loguru import logger
from pydantic import Field


class GatewayChat(BaseChatModel, ABC):
    """Abstract base class for Gateway API chat models."""

    api_url: str | None = Field(default=None, description="Base URL for the deployed API")
    model_name: str | None = Field(default=None, description="Model name to use")
    system_prompt: str = Field(default="You are a helpful assistant.", description="System prompt for the model")

    def __init__(self, **kwargs: Any):
        """
        Initialize the GatewayChat model.

        Args:
            api_url: Base URL for the deployed API.
            model_name: Name of the model to use.
            system_prompt: System prompt for the model.
        """
        # Check if api_url is provided in kwargs, otherwise get from environment
        if 'api_url' not in kwargs:
            api_url = os.getenv("CLOUD_RUN_API_URL")
            if not api_url:
                raise ValueError("CLOUD_RUN_API_URL environment variable is not set")
            kwargs['api_url'] = api_url

        # Call parent constructor with all fields
        super().__init__(**kwargs)

    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return the LLM type identifier."""
        pass

    @property
    @abstractmethod
    def _endpoint_path(self) -> str:
        """Return the API endpoint path for this LLM provider."""
        pass

    @property
    @abstractmethod
    def _stream_endpoint_path(self) -> str:
        """Return the streaming API endpoint path for this LLM provider."""
        pass

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for caching and tracing."""
        return {"api_url": self.api_url, "model_name": self.model_name}

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert list of messages to a single prompt string."""
        prompt = [f"System: {self.system_prompt}"]

        for message in messages:
            if isinstance(message, HumanMessage):
                prompt.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt.append(f"Assistant: {message.content}")
            else:
                prompt.append(f"{message.content}")
        return "\n".join(prompt)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response using the deployed API."""
        prompt = self._messages_to_prompt(messages)
        logger.debug(f"Generating response for prompt: {prompt}")

        # Get model from kwargs if provided, otherwise use default
        model_name = kwargs.get("model_name", self.model_name)

        try:
            # Prepare request payload
            payload = {"prompt": prompt}
            if model_name:  # Only add model_name if it's not empty
                payload["model_name"] = model_name

            response = requests.post(
                f"{self.api_url}{self._endpoint_path}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API Response: {result}")

                # Extract content from the response
                if "message" in result:
                    content = str(result["message"])
                else:
                    content = str(result)

                # Create ChatGeneration and return ChatResult
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                message = AIMessage(content=error_msg)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            logger.error(error_msg)
            message = AIMessage(content=error_msg)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response using the deployed API."""
        prompt = self._messages_to_prompt(messages)
        logger.debug(f"Starting Streaming response for prompt: {prompt}")

        # Get model from kwargs if provided, otherwise use default
        model_name = kwargs.get("model_name", self.model_name)

        try:
            # Prepare request payload
            payload = {"prompt": prompt}
            if model_name:  # Only add model_name if it's not empty
                payload["model_name"] = model_name

            response = requests.post(
                f"{self.api_url}{self._stream_endpoint_path}",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )

            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        chat_chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=chunk)
                        )
                        if run_manager:
                            run_manager.on_llm_new_token(chunk, chunk=chat_chunk)
                        yield chat_chunk
            else:
                error_text = f"API Error: {response.status_code} - {response.text}"
                logger.error(error_text)
                chat_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=error_text)
                )
                if run_manager:
                    run_manager.on_llm_new_token(error_text, chunk=chat_chunk)
                yield chat_chunk
        except Exception as e:
            error_text = f"Error calling API: {str(e)}"
            logger.error(error_text)
            chat_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=error_text)
            )
            if run_manager:
                run_manager.on_llm_new_token(error_text, chunk=chat_chunk)
            yield chat_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat response using the deployed API."""
        prompt = self._messages_to_prompt(messages)
        logger.debug(f"Starting Async streaming response for prompt: {prompt}")

        # Get model from kwargs if provided, otherwise use default
        model_name = kwargs.get("model_name", self.model_name)

        try:
            # Prepare request payload
            payload = {"prompt": prompt}
            if model_name:  # Only add model_name if it's not empty
                payload["model_name"] = model_name

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self._stream_endpoint_path}",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:

                    if response.status == 200:
                        async for chunk in response.content:
                            chunk_text = chunk.decode('utf-8')
                            if chunk_text:
                                # Create ChatGenerationChunk and yield it
                                chat_chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=chunk_text)
                                )
                                if run_manager:
                                    await run_manager.on_llm_new_token(
                                        chunk_text, chunk=chat_chunk
                                    )
                                yield chat_chunk
                    else:
                        error_text = f"API Error: {response.status}"
                        logger.error(error_text)
                        chat_chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=error_text)
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                error_text, chunk=chat_chunk
                            )
                        yield chat_chunk
        except Exception as e:
            error_text = f"Error calling API: {str(e)}"
            logger.error(error_text)
            chat_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=error_text)
            )
            if run_manager:
                await run_manager.on_llm_new_token(error_text, chunk=chat_chunk)
            yield chat_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate method for BaseChatModel."""
        # For now, fall back to sync generate
        # In production, you'd want to implement this with async HTTP calls
        return self._generate(messages, stop, None, **kwargs)


class GeminiChat(GatewayChat):
    """Custom LangChain Chat Model for your deployed Gemini API."""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", **kwargs: Any):
        """
        Initialize the GeminiChat model.

        Args:
            model_name: Name of the model to use.
        """
        # Set model_name in kwargs
        kwargs['model_name'] = model_name

        # Call parent constructor with all fields
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "gemini_chat"

    @property
    def _endpoint_path(self) -> str:
        return "/gemini"

    @property
    def _stream_endpoint_path(self) -> str:
        return "/gemini-stream"


class MistralChat(GatewayChat):
    """Custom LangChain Chat Model for your deployed Mistral API."""

    @property
    def _llm_type(self) -> str:
        return "mistral_chat"

    @property
    def _endpoint_path(self) -> str:
        return "/mistral"

    @property
    def _stream_endpoint_path(self) -> str:
        return "/mistral-stream"
