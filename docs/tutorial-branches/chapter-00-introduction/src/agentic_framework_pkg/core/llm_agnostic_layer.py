import litellm
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
from dotenv import load_dotenv

from .logger_config import get_logger

load_dotenv()

logger = get_logger(__name__)

class LLMServiceError(Exception):
    pass

class LLMAgnosticClient:
    DEFAULT_MODEL_ENV_VARS = {
        "streamlit": "STREAMLIT_DEFAULT_MODEL",
        "generic": "DEFAULT_LLM_MODEL",
    }

    def __init__(self):
        self.default_models = {
            purpose: os.getenv(env_var_name)
            for purpose, env_var_name in self.DEFAULT_MODEL_ENV_VARS.items()
        }

    async def agenerate_response(
        self,
        messages: List[Dict[str, str]],
        llm_purpose: str,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Any | AsyncGenerator[Any, None]:
        final_model = model
        if final_model is None:
            final_model = self.default_models.get(llm_purpose) or self.default_models.get("generic")
        if final_model is None:
            raise ValueError(f"No model specified and no default model found for purpose '{llm_purpose}' or generic default.")

        if final_model.startswith("ollama/"):
            kwargs["api_base"] = os.getenv("OLLAMA_API_BASE_URL")

        try:
            return await litellm.acompletion(
                model=final_model,
                messages=messages,
                stream=stream,
                **kwargs
            )
        except Exception as e:
            logger.error(f"LiteLLM completion error for model {final_model}: {e}", exc_info=True)
            raise LLMServiceError(f"LiteLLM completion error for model {final_model}: {e}") from e
