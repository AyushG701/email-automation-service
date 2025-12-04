from functools import lru_cache
from langchain_openai import ChatOpenAI
from openai import OpenAI
# from configs.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE, OPENAI_API_KEY
from config import AppConfig as Settings
import structlog

logger = structlog.get_logger(__name__)


class OpenAIClientManager:
    """Singleton manager for OpenAI clients to avoid multiple TCP connections."""

    _openai_client = None
    _langchain_clients = {}

    @classmethod
    def get_openai_client(cls) -> OpenAI:
        """Get a singleton OpenAI client for direct API calls."""
        if cls._openai_client is None:
            if not Settings.OPENAI_API_KEY:
                raise ValueError(
                    "OpenAI API key is not set. Please check your .env file.")

            cls._openai_client = OpenAI(api_key=Settings.OPENAI_API_KEY)
            try:
                # Test connection
                cls._openai_client.models.list()
                logger.info("Successfully connected to OpenAI API")
            except Exception as e:
                logger.error(f"Failed to connect to OpenAI API: {str(e)}")
                raise

        return cls._openai_client

    @classmethod
    def get_langchain_client(cls, model: str = Settings.DEFAULT_MODEL, temperature: float = Settings.DEFAULT_TEMPERATURE) -> ChatOpenAI:
        """Get a cached LangChain client for the specified model and temperature."""
        client_key = f"{model}_{temperature}"

        if client_key not in cls._langchain_clients:
            if not Settings.OPENAI_API_KEY:
                raise ValueError(
                    "OpenAI API key is not set. Please check your .env file.")

            cls._langchain_clients[client_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=Settings.OPENAI_API_KEY
            )
            logger.info(
                f"Created new LangChain client for {model} with temperature {temperature}")

        return cls._langchain_clients[client_key]

# Backward compatibility functions


def get_llm(model=Settings.DEFAULT_MODEL, temperature=Settings.DEFAULT_TEMPERATURE):
    """Initialize and return chatgpt instance (backward compatible)."""
    return OpenAIClientManager.get_langchain_client(model, temperature)


def get_openai_client():
    """Get the singleton OpenAI client."""
    return OpenAIClientManager.get_openai_client()
