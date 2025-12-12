# LangGraph Text Analysis Workflow

A comprehensive LangGraph implementation demonstrating a multi-node workflow for text analysis using Ollama models.

## Project Structure

```
backend/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── models.py          # Model configuration
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py            # State definition
│   │   ├── nodes.py            # Node implementations
│   │   └── workflow.py         # Graph definition
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Helper functions
├── main.py                     # Main entry point
├── requirements.txt
└── README.md
```

## Features

- **State Management**: TypedDict-based state with 3 distinct fields
- **Multi-Node Workflow**: 2-node processing pipeline
  - Input Processor: Analyzes text and counts words
  - Summarizer: Generates summary and sentiment analysis
- **Ollama Integration**: Configurable model selection
- **Memory Persistence**: Built-in checkpointing with MemorySaver
- **Comprehensive Logging**: Detailed execution tracking

## Use Case

This workflow implements a text analysis system:
1. User provides input text
2. Node 1 processes the input and calculates metadata (word count)
3. Node 2 reads the text and metadata to generate:
   - A concise summary
   - Sentiment analysis

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running:
```bash
ollama serve
```

3. Pull required models:
```bash
ollama pull llama3.2
```

4. Create `config.json` file for default model configuration:
```json
{
  "default_model": "qwen3:1.7b"
}
```

5. Create `.env` file for other configurations (optional):
```bash
cp .env.example .env
```

## Usage

Run the workflow:
```bash
python main.py
```

Or import and use programmatically:
```python
from src.graph.workflow import create_workflow

# Create workflow with default model
workflow = create_workflow()

# Or specify a model
workflow = create_workflow(model_name="llama3.2")

# Run the workflow
result = workflow.invoke({
    "input_text": "Your text here..."
})
```

## Configuration

The application uses `config.json` for model and embedding configuration. Create this file in the `backend/` directory:

```json
{
  "default_model": "qwen3:1.7b",
  "embedding_provider": "ollama",
  "embedding_model": "nomic-embed-text"
}
```

### Configuration Keys

- **`default_model`** (string): The default Ollama model to use for chat responses. Examples: "llama3.2", "qwen3:1.7b", "mistral". This model is loaded when no specific model is provided.

- **`embedding_provider`** (string): The provider for generating text embeddings. Options:
  - `"ollama"` (default): Use local Ollama models for embeddings
  - `"azure"`: Use Azure OpenAI embeddings service

- **`embedding_model`** (string): The specific embedding model to use, depending on the provider:
  - For Ollama: Model names like "nomic-embed-text", "mxbai-embed-large", "all-minilm"
  - For Azure: The deployment name, e.g., "text-embedding-3-small_mimi"

### Environment Variables (.env)

Additional configuration is handled via environment variables in a `.env` file:

- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key (required for Azure embeddings)
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_VERSION`: API version (default: 2024-12-01-preview)
- `AZURE_EMBEDDING_DEPLOYMENT`: Azure embedding deployment name
- `LANGSMITH_*`: LangSmith tracing configuration (optional)

## Requirements

- Python 3.11+
- Ollama installed and running
- At least one Ollama model pulled
