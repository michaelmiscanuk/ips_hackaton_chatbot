# Quick Start Guide - ChromaDB with Hybrid Search

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (1 min)

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Configure Environment (1 min)

Create `backend/.env` file:

```env
# Required - Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Optional - For better results
COHERE_API_KEY=your-cohere-key
TRANSLATOR_TEXT_SUBSCRIPTION_KEY=your-translator-key
TRANSLATOR_TEXT_REGION=westeurope
TRANSLATOR_TEXT_ENDPOINT=https://api.cognitive.microsofttranslator.com/
```

### Step 3: Prepare Your Data (1 min)

Ensure `backend/data/sample0.csv` exists with this structure:

```csv
subject,body,answer,type,language
"Password Reset","How to reset password","Go to settings...",FAQ,en
"Account Login","Cannot login","Check your credentials...",Support,en
```

### Step 4: Setup ChromaDB (2 min)

**Windows**:
```bash
setup_all.bat
```

**Linux/Mac**:
```bash
python setup_chromadb.py
```

This will:
- Load CSV data
- Translate text (if configured)
- Create embeddings
- Build ChromaDB
- Test the setup

### Step 5: Run Your Application (30 sec)

```bash
cd backend
python -m uvicorn api:app --reload
```

Or use:
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

## ✅ You're Done!

Visit: http://localhost:8000/docs

Test the `/api/chat` endpoint with:
```json
{
  "message": "How do I reset my password?"
}
```

## 🧪 Test Hybrid Search

```bash
python test_chromadb.py "your search query"
```

Example:
```bash
python test_chromadb.py "How do I reset my password?"
```

## 📊 View Results

After testing, check:
- Terminal logs for search scores
- `backend/data/search_comparison.xlsx` for detailed comparison

## 🔄 Update Data

When you add/change data in `sample0.csv`:

```bash
python update_chromadb.py
```

## 📚 Learn More

- Full documentation: `CHROMADB_HYBRID_SEARCH_README.md`
- Implementation details: `CHROMADB_INTEGRATION_SUMMARY.md`

## 🆘 Need Help?

**ChromaDB not found?**
```bash
python setup_chromadb.py
```

**Dependencies missing?**
```bash
pip install -r requirements.txt
```

**Check everything is working?**
```bash
python test_chromadb.py "test query"
```

## 🎯 What You Get

✨ **Hybrid Search** - Combines semantic understanding + keyword matching
🔍 **Better Results** - Cohere reranking for improved relevance  
🌍 **Multilingual** - Works with multiple languages
📈 **Scalable** - Handles growing datasets efficiently
🔄 **Fallback** - Gracefully degrades if optional components unavailable

## 🏃 Next Steps

1. ✅ Test with your actual queries
2. 📊 Review search comparisons
3. ⚙️ Tune parameters if needed
4. 🚀 Deploy to production

Happy searching! 🎉
