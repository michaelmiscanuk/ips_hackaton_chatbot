"""
Node implementations for the LangGraph workflow

This module contains the node functions that perform the actual
processing in the workflow. Each node receives the current state
and returns updates to it.
"""

import os
import sys
import logging
from typing import Dict, Any, List
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from .state import ChatState
from ..config.models import get_model, get_langchain_azure_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Add data directory to path for importing chromadb_manager
sys.path.insert(0, str(DATA_DIR))

try:
    from chromadb_manager import hybrid_search, cohere_rerank, get_chromadb_collection

    HYBRID_SEARCH_AVAILABLE = True
    logger.info("✅ Hybrid search module loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Hybrid search not available: {e}")
    HYBRID_SEARCH_AVAILABLE = False


def retrieve(state: ChatState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from ChromaDB using hybrid search.

    This function implements a sophisticated retrieval strategy:
    1. Hybrid search (semantic + BM25) to get diverse results
    2. Cohere reranking for improved relevance
    3. Fallback to simple semantic search if hybrid search unavailable
    """
    logger.info("NODE: Retrieve - Starting")

    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages found in state")
        return {"context": []}

    last_message = messages[-1]
    query = last_message.content

    logger.info(f"Querying ChromaDB for: {query}")

    try:
        if not CHROMA_DB_DIR.exists():
            logger.warning(f"ChromaDB directory not found at {CHROMA_DB_DIR}")
            return {"context": []}

        # Try hybrid search first
        if HYBRID_SEARCH_AVAILABLE:
            try:
                logger.info("Using hybrid search (semantic + BM25)")

                # Get ChromaDB collection
                collection = get_chromadb_collection()

                # Perform hybrid search (semantic + BM25)
                hybrid_results = hybrid_search(collection, query, n_results=30)

                if hybrid_results:
                    # Convert hybrid results to Document objects for reranking
                    hybrid_docs = [
                        Document(
                            page_content=result["document"], metadata=result["metadata"]
                        )
                        for result in hybrid_results
                    ]

                    # Apply Cohere reranking to get top results
                    logger.info("Applying Cohere reranking")
                    reranked_results = cohere_rerank(query, hybrid_docs, top_n=5)

                    # Extract context from reranked results
                    context = []
                    for doc, res in reranked_results:
                        context.append(doc.page_content)
                        if res:
                            logger.info(
                                f"  - Relevance score: {res.relevance_score:.4f}"
                            )

                    logger.info(
                        f"Retrieved {len(context)} documents via hybrid search + reranking"
                    )
                    return {"context": context}

            except Exception as e:
                logger.error(f"Hybrid search failed, falling back to semantic: {e}")

        # Fallback to simple semantic search
        logger.info("Using simple semantic search")
        embedding_model = get_langchain_azure_embedding_model()
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR), embedding_function=embedding_model
        )

        # Search for top 5 similar documents
        docs = vectorstore.similarity_search(query, k=5)
        context = [doc.page_content for doc in docs]

        logger.info(f"Retrieved {len(context)} documents via semantic search")
        return {"context": context}

    except Exception as e:
        logger.error(f"Error in retrieve node: {e}")
        return {"context": []}


def generate(state: ChatState) -> Dict[str, Any]:
    """
    Generate a response using the LLM, context, and conversation history.
    """
    logger.info("NODE: Generate - Starting")

    messages = state.get("messages", [])
    context = state.get("context", [])

    # Format context
    context_str = "\n\n".join(context) if context else "No relevant information found."

    system_prompt = f"""You are a helpful customer support assistant. 
Use the following context to answer the user's question. 
If the answer is not in the context, politely say that you don't have the answer and suggest contacting human support at support@example.com.
Keep your answers concise and helpful.

Context:
{context_str}
"""

    # Prepare messages for LLM
    # We prepend the system prompt to the conversation history
    prompt_messages = [SystemMessage(content=system_prompt)] + messages

    try:
        model = get_model(temperature=0.7)
        response = model.invoke(prompt_messages)

        logger.info("Response generated")
        return {"messages": [response]}

    except Exception as e:
        logger.error(f"Error in generate node: {e}")
        return {
            "messages": [
                AIMessage(
                    content="I apologize, but I encountered an error while processing your request."
                )
            ]
        }
