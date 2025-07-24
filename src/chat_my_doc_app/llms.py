from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from loguru import logger
from pydantic import Field


class CloudRunLLM(LLM):
    api_url: str = Field(...)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        return "cloud_run_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        import requests
        
        # Get model from kwargs if provided, otherwise use default
        model_name = kwargs.get("model_name", "gemini-2.0-flash-lite")
        
        try:
            response = requests.post(
                f"{self.api_url}/gemini-model",
                params={"prompt": prompt, "model_name": model_name}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Type of response {type(response)} & Response from API: {result}")
                
                # Handle different possible response formats
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    # Check if 'content' field exists and extract it directly
                    if "message" in result:
                        return str(result["message"]["content"])
                    return str(result)
                else:
                    return str(result)
            else:
                return f"API Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"