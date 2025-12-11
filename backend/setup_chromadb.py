"""
Setup script to initialize ChromaDB with CSV data

This script loads data from sample0.csv and creates a ChromaDB collection
with hybrid search capabilities.

Usage:
    python setup_chromadb.py
"""

import asyncio
import sys
from pathlib import Path

# Add data directory to path
sys.path.insert(0, str(Path(__file__).parent / "data"))

from chromadb_manager import upsert_documents_to_chromadb


async def main():
    """Main setup function."""
    print("=" * 70)
    print("ChromaDB Setup Script")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Load data from backend/data/sample0.csv")
    print("  2. Translate text to English (if translator configured)")
    print("  3. Create ChromaDB with embeddings")
    print("  4. Test hybrid search capabilities")
    print()
    print("=" * 70)
    print()

    try:
        # Create and populate ChromaDB
        collection = await upsert_documents_to_chromadb(
            deployment="text-embedding-3-large__test1",
            collection_name="chatbot_collection",
        )

        print()
        print("=" * 70)
        print("✅ ChromaDB Setup Complete!")
        print("=" * 70)
        print()
        print("You can now run your application with:")
        print("  cd backend")
        print("  python -m uvicorn api:app --reload")
        print()
        print("Or use the start script:")
        print("  cd backend")
        print("  ./start.bat  (Windows)")
        print("  ./start.sh   (Linux/Mac)")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Setup Failed: {e}")
        print("=" * 70)
        print()
        print("Make sure you have:")
        print("  1. Created backend/data/sample0.csv")
        print("  2. Configured Azure OpenAI credentials in .env")
        print("  3. (Optional) Configured Azure Translator for translation")
        print("  4. (Optional) Configured Cohere API for reranking")
        print()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
