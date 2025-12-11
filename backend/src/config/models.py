"""
Model configuration and initialization

This module provides a centralized way to configure and initialize
Ollama models with different parameters.
"""

# pylint: disable=import-error

import os
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import AIMessage


class MockChatOllama:
    """Mock ChatOllama for testing without Ollama"""

    def invoke(self, messages):
        return AIMessage(content="Mock response from Chatbot")


class ModelConfig:
    """Configuration class for Ollama models"""

    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        num_ctx: int = 2048,
        top_p: float = 0.9,
        top_k: int = 40,
    ):
        """
        Initialize model configuration

        Args:
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature (0.0 to 1.0)
            base_url: Base URL for Ollama API (defaults to localhost)
            num_ctx: Context window size
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        self.model_name = model_name
        self.temperature = temperature
        # Use OLLAMA_HOST from Railway environment, fallback to localhost for dev
        self.base_url = base_url or os.getenv(
            "OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.num_ctx = num_ctx
        self.top_p = top_p
        self.top_k = top_k

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "num_ctx": self.num_ctx,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


def get_model(
    model_name: Optional[str] = None, temperature: float = 0.7, **kwargs
) -> ChatOllama:
    """
    Get a configured ChatOllama instance

    This is the main function to get a model instance. It supports
    passing a model name and configuration parameters.

    Args:
        model_name: Name of the Ollama model (defaults to env var or llama3.2)
        temperature: Sampling temperature
        **kwargs: Additional parameters for ModelConfig

    Returns:
        Configured ChatOllama instance

    Example:
        >>> model = get_model("llama3.2", temperature=0.5)
        >>> model = get_model()  # Uses default configuration
    """
    if model_name is None:
        model_name = os.getenv("DEFAULT_MODEL", "llama3.2")

    config = ModelConfig(model_name=model_name, temperature=temperature, **kwargs)

    # Try to use Ollama, but fall back to mock if not available
    try:
        model = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
            num_ctx=config.num_ctx,
            num_gpu=0,  # Force CPU-only mode (0 = disable GPU)
            # Additional Ollama-specific parameters
            format="",  # Empty string for regular text generation
        )
        # Test if Ollama is running by trying a simple invoke
        model.invoke([{"role": "user", "content": "test"}])
        return model
    except Exception as e:
        print(f"Ollama not available, using mock model. Error: {str(e)}")
        return MockChatOllama()


def get_langchain_azure_embedding_model(deployment_name="text-embedding-3-small_mimi"):
    """Get a LangChain AzureOpenAIEmbeddings instance with standard configuration.

    Args:
        deployment_name (str): The name of the Azure deployment (e.g., "text-embedding-3-small_mimi")

    Returns:
        AzureOpenAIEmbeddings: Configured embedding model instance
    """

    return AzureOpenAIEmbeddings(
        model="text-embedding-3-small",  # The actual model name
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        deployment=deployment_name,  # Your Azure deployment name
    )
