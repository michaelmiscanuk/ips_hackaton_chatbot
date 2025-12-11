# ChromaDB with Hybrid Search Integration

This document explains how the ChromaDB integration with hybrid search works in your chatbot application.

## Overview

Your application now uses a sophisticated retrieval system that combines:
1. **Semantic Search** - Using Azure OpenAI embeddings for meaning-based retrieval
2. **BM25 Search** - Keyword-based search for exact matches
3. **Cohere Reranking** - Advanced reranking for improved relevance

## Architecture

### Data Flow

```
CSV Data (sample0.csv)
    ↓
Translation (Azure Translator)
    ↓
Embedding (Azure OpenAI)
    ↓
ChromaDB Storage
    ↓
Query → Hybrid Search (Semantic + BM25)
    ↓
Cohere Reranking
    ↓
Top Results → LLM
```

### Components

1. **chromadb_manager.py** - Core module handling:
   - CSV data loading
   - Text translation
   - ChromaDB operations
   - Hybrid search implementation
   - Cohere reranking

2. **nodes.py (modified)** - Integration into LangGraph workflow:
   - Uses hybrid search in retrieve node
   - Falls back to semantic search if hybrid unavailable
   - Applies Cohere reranking for best results

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `backend` directory with:

```env
# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Translator (Optional - for translation)
TRANSLATOR_TEXT_SUBSCRIPTION_KEY=your-translator-key
TRANSLATOR_TEXT_REGION=westeurope
TRANSLATOR_TEXT_ENDPOINT=https://api.cognitive.microsofttranslator.com/

# Cohere (Optional - for reranking)
COHERE_API_KEY=your-cohere-api-key
```

### 3. Prepare Your Data

Ensure `backend/data/sample0.csv` exists with columns:
- `subject` - The subject/title
- `body` - The main content
- `answer` - The answer/response
- `type` - Document type
- `language` - Source language

Example:
```csv
subject,body,answer,type,language
"Password Reset","How to reset password","Click Forgot Password...",FAQ,en
```

### 4. Initialize ChromaDB

Run the setup script:

```bash
cd backend
python setup_chromadb.py
```

This will:
- Load data from CSV
- Translate text (if translator configured)
- Generate embeddings
- Create ChromaDB collection
- Test the setup

### 5. Run the Application

```bash
cd backend
python -m uvicorn api:app --reload
```

Or use the start script:
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

## How Hybrid Search Works

### Search Process

1. **User Query** → "How do I reset my password?"

2. **Semantic Search** (85% weight)
   - Converts query to embedding vector
   - Finds semantically similar documents
   - Returns top N results with similarity scores

3. **BM25 Search** (15% weight)
   - Tokenizes query and documents
   - Calculates keyword-based relevance
   - Returns top N results with BM25 scores

4. **Score Combination**
   ```python
   final_score = (semantic_score * 0.85) + (bm25_score * 0.15)
   ```

5. **Cohere Reranking**
   - Takes combined results
   - Uses Cohere's multilingual reranker
   - Returns top K most relevant documents

6. **Context to LLM**
   - Top K documents become context
   - Fed to LLM for response generation

### Benefits

- **Better Recall**: Semantic search finds conceptually similar content
- **Better Precision**: BM25 ensures exact keyword matches aren't missed
- **Improved Ranking**: Cohere reranking optimizes final order
- **Multilingual**: Works with multiple languages

## Testing the Integration

### Test Hybrid Search

```bash
cd backend/data
python chromadb_manager.py
```

This runs a complete test including:
- Pure semantic search
- Hybrid search
- Cohere reranking
- Excel comparison output

### Check Logs

The application logs show which search method is used:

```
✅ Hybrid search module loaded successfully
Using hybrid search (semantic + BM25)
Applying Cohere reranking
Retrieved 5 documents via hybrid search + reranking
```

### Excel Comparison

After testing, check `backend/data/search_comparison.xlsx` to see:
- Semantic vs Hybrid vs Reranked rankings
- Score comparisons
- Which method found which results

## Fallback Behavior

If hybrid search is unavailable (missing dependencies, etc.), the system automatically falls back to simple semantic search:

```
⚠️ Hybrid search not available: ...
Using simple semantic search
Retrieved 5 documents via semantic search
```

## Configuration Options

### In chromadb_manager.py

```python
# Number of results from hybrid search
n_results = 30  # Get 30 candidates

# Top results after reranking
top_n = 5  # Keep top 5

# Search weights
SEMANTIC_WEIGHT = 0.85  # 85% semantic
BM25_WEIGHT = 0.15      # 15% BM25
```

### In nodes.py

```python
# Number of candidates for hybrid search
hybrid_results = hybrid_search(collection, query, n_results=30)

# Number of final results after reranking
reranked_results = cohere_rerank(query, hybrid_docs, top_n=5)
```

## Troubleshooting

### ChromaDB not found
```
ChromaDB directory not found at ...
```
**Solution**: Run `python setup_chromadb.py` first

### Hybrid search failed
```
Hybrid search failed, falling back to semantic: ...
```
**Solution**: Check that `rank-bm25` is installed: `pip install rank-bm25`

### Translation not working
```
Translation credentials not found, skipping translation
```
**Solution**: Add Azure Translator credentials to `.env` (optional)

### Cohere reranking skipped
```
COHERE_API_KEY not found, skipping reranking
```
**Solution**: Add Cohere API key to `.env` (optional)

## Performance Optimization

### For Large Datasets

1. **Batch Processing**: Data loading uses batches
   ```python
   batch_size = 10  # Adjust based on memory
   ```

2. **Token Limits**: Long documents are automatically split
   ```python
   MAX_TOKENS = 8190  # Azure OpenAI limit
   ```

3. **Caching**: ChromaDB persists to disk
   ```python
   CHROMA_DB_DIR = "backend/data/chroma_db"
   ```

### For Production

1. Enable persistence: Already enabled via `PersistentClient`
2. Use appropriate k values: Balance speed vs quality
3. Monitor API costs: Azure OpenAI, Cohere charges apply
4. Consider async operations: For high-traffic scenarios

## API Endpoints

Your chatbot API remains the same:

```http
POST /api/chat
Content-Type: application/json

{
  "message": "How do I reset my password?",
  "thread_id": "optional-conversation-id"
}
```

Response:
```json
{
  "response": "To reset your password, click the 'Forgot Password' link...",
  "thread_id": "conversation-id",
  "success": true
}
```

The hybrid search happens transparently in the backend.

## Comparison with Previous Version

### Before (Simple Semantic Search)
- Single embedding-based search
- Top K results directly to LLM
- No keyword matching
- No reranking

### After (Hybrid Search + Reranking)
- Combined semantic + BM25 search
- Weighted score combination
- Cohere reranking for relevance
- Better precision and recall

## Next Steps

1. **Test with Your Data**: Run setup and test queries
2. **Tune Parameters**: Adjust weights, k values based on results
3. **Monitor Performance**: Check logs and Excel comparisons
4. **Optimize Costs**: Consider caching, batch processing
5. **Scale Up**: Add more data, test with production traffic

## Support

For issues or questions:
1. Check the logs in the terminal
2. Review the Excel comparison files
3. Verify environment variables
4. Check dependencies with `pip list`

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Cohere Rerank](https://docs.cohere.com/docs/reranking)
- [Azure OpenAI Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/embeddings)
