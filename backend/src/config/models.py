"""
Model configuration and initialization

This module provides a centralized way to configure and initialize
Ollama models with different parameters.
"""

# pylint: disable=import-error

import os
import json
from pathlib import Path
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import AIMessage

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None


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
        model_name: Name of the Ollama model (defaults to config.json or llama3.2)
        temperature: Sampling temperature
        **kwargs: Additional parameters for ModelConfig

    Returns:
        Configured ChatOllama instance

    Example:
        >>> model = get_model("llama3.2", temperature=0.5)
        >>> model = get_model()  # Uses default configuration
    """
    if model_name is None:
        config_path = Path(__file__).parent.parent.parent / "config.json"
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            model_name = config.get("default_model", "llama3.2")
        except (FileNotFoundError, json.JSONDecodeError):
            model_name = "llama3.2"

    config = ModelConfig(model_name=model_name, temperature=temperature, **kwargs)

    # Try to use Ollama, but fall back to mock if not available
    try:
        model = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
            num_ctx=config.num_ctx,
            # num_gpu=-1 would use all GPUs (default behavior when not specified)
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


def _load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_embedding_model_name():
    """
    Get the current embedding model name for naming ChromaDB directories.

    Returns:
        str: Sanitized model name suitable for directory names

    Example:
        >>> get_embedding_model_name()
        'nomic-embed-text'  # for Ollama
        'text-embedding-3-small_mimi'  # for Azure
    """
    config = _load_config()

    # Env vars take precedence, then config.json, then default
    provider = os.getenv(
        "EMBEDDING_PROVIDER", config.get("embedding_provider", "ollama")
    ).lower()

    if provider == "azure":
        model_name = os.getenv(
            "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small_mimi"
        )
    else:
        model_name = os.getenv(
            "EMBEDDING_MODEL", config.get("embedding_model", "nomic-embed-text")
        )

    # Sanitize model name for use in paths (replace unsafe characters)
    return model_name.replace("/", "-").replace("\\", "-").replace(":", "-")


def get_embedding_model():
    """
    Get the configured embedding model based on environment variables.

    Returns either Ollama or Azure embeddings based on EMBEDDING_PROVIDER setting.
    Default is Ollama with nomic-embed-text model.

    Environment Variables:
        EMBEDDING_PROVIDER: "ollama" (default) or "azure"
        EMBEDDING_MODEL: Model name (default: "nomic-embed-text" for Ollama)
        AZURE_EMBEDDING_DEPLOYMENT: Azure deployment name (when using Azure)
        OLLAMA_BASE_URL: Ollama API URL (default: http://localhost:11434)

    Returns:
        Embedding model instance (OllamaEmbeddings or AzureOpenAIEmbeddings)

    Example:
        >>> embeddings = get_embedding_model()
        >>> vectors = embeddings.embed_documents(["text1", "text2"])
    """
    config = _load_config()

    # Env vars take precedence, then config.json, then default
    provider = os.getenv(
        "EMBEDDING_PROVIDER", config.get("embedding_provider", "ollama")
    ).lower()

    if provider == "azure":
        # Use Azure OpenAI embeddings
        deployment_name = os.getenv(
            "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small_mimi"
        )
        return get_langchain_azure_embedding_model(deployment_name=deployment_name)
    else:
        # Use Ollama embeddings (default)
        if OllamaEmbeddings is None:
            raise ImportError(
                "OllamaEmbeddings not available. Install with: pip install langchain-ollama"
            )

        model_name = os.getenv(
            "EMBEDDING_MODEL", config.get("embedding_model", "nomic-embed-text")
        )
        base_url = os.getenv(
            "OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

        # Configure with timeout and keep-alive for large batch processing
        return OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
            # Increase timeout for large batches (default is 30s)
            # Large batches (200 docs) may take 30-60s to process
        )
