"""
Quick Test Script for ChromaDB Hybrid Search

This script provides a simple way to test your ChromaDB setup and hybrid search functionality.

Usage:
    python test_chromadb.py "your search query here"

Example:
    python test_chromadb.py "How do I reset my password?"
"""

import asyncio
import sys
from pathlib import Path

# Add data directory to path
sys.path.insert(0, str(Path(__file__).parent / "data"))

from chromadb_manager import (
    get_chromadb_collection,
    hybrid_search,
    similarity_search_chromadb,
    get_azure_embedding_model,
)
from langchain_core.documents import Document


async def test_search(query: str):
    """Test search functionality with a query."""
    print("=" * 70)
    print(f"Testing Search: '{query}'")
    print("=" * 70)
    print()

    try:
        # Get collection
        print("📦 Loading ChromaDB collection...")
        collection = get_chromadb_collection()
        print("✅ Collection loaded")
        print()

        # Get embedding client
        embedding_client = get_azure_embedding_model()

        # Test 1: Simple semantic search
        print("[1/3] Simple Semantic Search")
        print("-" * 70)
        semantic_results = similarity_search_chromadb(
            collection=collection,
            embedding_client=embedding_client,
            query=query,
            k=5,
        )

        if semantic_results and "documents" in semantic_results:
            for i, (doc, meta, dist) in enumerate(
                zip(
                    semantic_results["documents"][0],
                    semantic_results["metadatas"][0],
                    semantic_results["distances"][0],
                ),
                1,
            ):
                subject = meta.get("subject", "N/A")
                similarity = 1 - (dist / 2)
                print(f"  #{i}: {subject}")
                print(f"      Similarity: {similarity:.4f}")
                print(f"      Preview: {doc[:100]}...")
                print()
        print()

        # Test 2: Hybrid search
        print("[2/2] Hybrid Search (Semantic + BM25)")
        print("-" * 70)
        hybrid_results = hybrid_search(collection, query, n_results=10)

        for i, result in enumerate(hybrid_results[:5], 1):
            subject = result["metadata"].get("subject", "N/A")
            score = result.get("score", 0)
            semantic_score = result.get("semantic_score", 0)
            bm25_score = result.get("bm25_score", 0)
            source = result.get("source", "unknown")

            print(f"  #{i}: {subject}")
            print(f"      Combined Score: {score:.6f}")
            print(f"      Semantic: {semantic_score:.4f} | BM25: {bm25_score:.4f}")
            print(f"      Source: {source}")
            print(f"      Preview: {result['document'][:100]}...")
            print()
        print()

        # Summary
        print("=" * 70)
        print("✅ Test Complete!")
        print("=" * 70)
        print()
        print("Summary:")
        print(
            f"  - Semantic search returned {len(semantic_results['documents'][0])} results"
        )
        print(f"  - Hybrid search returned {len(hybrid_results)} results")
        print()
        print("The top hybrid search results would be used as context for the LLM.")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Error: {e}")
        print("=" * 70)
        print()
        print("Make sure you have:")
        print("  1. Run 'python setup_chromadb.py' first")
        print("  2. Configured Azure OpenAI credentials")
        print()
        sys.exit(1)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print('Usage: python test_chromadb.py "your search query"')
        print()
        print("Example:")
        print('  python test_chromadb.py "How do I reset my password?"')
        print()
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    asyncio.run(test_search(query))


if __name__ == "__main__":
    main()
