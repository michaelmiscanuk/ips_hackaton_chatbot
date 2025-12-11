"""
FastAPI REST API for LangGraph Chatbot Backend

This module provides REST API endpoints for the chatbot workflow.
Designed to be deployed on Render.com and accessed by the frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import sys
import os
import uuid
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot API",
    description="LangGraph-powered chatbot with RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS - Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",  # Local development
        "http://127.0.0.1:5000",
        "https://summarizer-agent-langgraph-ufot.vercel.app",  # Production Vercel frontend
        "https://*.vercel.app",  # All Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat"""

    message: str = Field(..., min_length=1, description="User message")
    thread_id: Optional[str] = Field(None, description="Conversation ID for history")


class ChatResponse(BaseModel):
    """Response model for chat"""

    response: str
    thread_id: str
    success: bool = True


# API Endpoints
@app.get("/")
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "status": "healthy",
        "message": "Chatbot API is running. Visit /docs for API documentation.",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {"status": "healthy", "message": "API is operational"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat with the bot

    Args:
        request: ChatRequest with message and optional thread_id

    Returns:
        ChatResponse with AI response
    """
    try:
        logger.info(f"Received chat request. Message: {request.message[:50]}...")

        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        logger.info(f"Using thread_id: {thread_id}")

        # Run workflow
        from src.graph.workflow import run_workflow

        logger.info(f"Running workflow for thread {thread_id}")
        try:
            result = run_workflow(input_text=request.message, thread_id=thread_id)
            logger.info("Workflow execution completed")
        except Exception as wf_error:
            logger.error(f"Workflow execution failed: {wf_error}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Workflow error: {str(wf_error)}"
            )

        # Get the last message (AI response)
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        response_text = (
            last_message.content if last_message else "No response generated."
        )

        logger.info(f"Response generated: {response_text[:50]}...")

        logger.info("Chat completed successfully")
        return {
            "response": response_text,
            "thread_id": thread_id,
            "success": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing request: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error: %s" % str(e)
        )


# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    reload_mode = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run(
        "api:app", host="0.0.0.0", port=8000, reload=reload_mode, log_level="info"
    )
