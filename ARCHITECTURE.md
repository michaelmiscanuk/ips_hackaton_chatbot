# Application Architecture

## Overview

This is a **full-stack RAG (Retrieval-Augmented Generation) chatbot application** built with LangGraph, featuring a Python backend and Flask frontend. The application enables intelligent question-answering using document retrieval and LLM-based generation.

**Last Updated:** December 11, 2025

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Flask Frontend (app.py)                      │ │
│  │  - Web UI (templates/index.html)                         │ │
│  │  - Static assets (CSS/JS)                                │ │
│  │  - API proxy layer                                       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BACKEND API LAYER                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              FastAPI Server (api.py)                      │ │
│  │  - /api/chat endpoint                                    │ │
│  │  - CORS middleware                                       │ │
│  │  - Request/Response validation                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ invoke
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH WORKFLOW LAYER                    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                 LangGraph Workflow                        │ │
│  │                                                           │ │
│  │     START → retrieve → generate → END                    │ │
│  │                                                           │ │
│  │  State: ChatState (messages, context)                    │ │
│  │  Memory: MemorySaver checkpointer                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                 │                              │
                 │ retrieval                    │ generation
                 ▼                              ▼
┌──────────────────────────────┐  ┌─────────────────────────────┐
│     DATA LAYER               │  │      LLM LAYER              │
│                              │  │                             │
│  ┌────────────────────────┐  │  │  ┌───────────────────────┐ │
│  │   ChromaDB             │  │  │  │   Ollama LLM          │ │
│  │  - Vector storage      │  │  │  │  - Local inference    │ │
│  │  - Hybrid search       │  │  │  │  - Model presets      │ │
│  │  - Document chunks     │  │  │  │  - Temperature config │ │
│  └────────────────────────┘  │  │  └───────────────────────┘ │
└──────────────────────────────┘  └─────────────────────────────┘
```

---

## Component Breakdown

### 1. Frontend (Flask Application)

**Location:** `frontend/`

**Key Files:**
- `app.py` - Flask application server
- `templates/` - HTML templates (Jinja2)
- `static/` - CSS and JavaScript assets

**Responsibilities:**
- Render web UI for user interactions
- Proxy API requests to backend
- Handle session management
- Display chat history and responses
- Error handling and user feedback

**Deployment:**
- **Platform:** Vercel
- **URL:** `https://summarizer-agent-langgraph-ufot.vercel.app`

---

### 2. Backend API (FastAPI)

**Location:** `backend/api.py`

**Key Features:**
- RESTful API endpoints
- CORS configuration for frontend
- Request/response validation with Pydantic
- Health check endpoints
- Error handling and logging

**Main Endpoints:**
- `GET /` - API health check
- `GET /health` - Monitoring endpoint
- `POST /api/chat` - Chat interaction

**Request Flow:**
```
1. Receive ChatRequest (message, thread_id)
2. Generate/validate thread_id for conversation history
3. Invoke LangGraph workflow
4. Extract AI response from workflow result
5. Return ChatResponse with response text
```

---

### 3. LangGraph Workflow Engine

**Location:** `backend/src/graph/`

**Core Components:**

#### State Management (`state.py`)
- `ChatState` - TypedDict with:
  - `messages` - List of conversation messages
  - `context` - Retrieved document context

#### Workflow Nodes (`nodes.py`)
- **retrieve** - Searches ChromaDB for relevant documents
- **generate** - Uses LLM to generate contextual response

#### Workflow Orchestration (`workflow.py`)
- Graph construction with StateGraph
- Node connections and edges
- MemorySaver checkpointer for persistence
- Thread-based conversation history

**Workflow Execution:**
```
Input: User message + thread_id
  ↓
retrieve node:
  - Query ChromaDB with user message
  - Retrieve top-k relevant documents
  - Update state with context
  ↓
generate node:
  - Combine user message + retrieved context
  - Generate response using Ollama LLM
  - Update state with AI response
  ↓
Output: Final state with complete conversation
```

---

### 4. Data Layer (ChromaDB)

**Location:** `backend/data/`

**Key Files:**
- `chromadb_manager.py` - ChromaDB interface
- `ingest_data.py` - Document ingestion
- `chroma_db/` - Vector database storage
- Sample documents (`sample*.txt`)

**Features:**
- Vector embeddings for semantic search
- Hybrid search capabilities
- Document chunk management
- Persistent storage

**Data Flow:**
```
Document Ingestion:
  Raw Text → Chunking → Embedding → ChromaDB Storage

Retrieval:
  User Query → Embedding → Vector Search → Top-K Docs
```

---

### 5. LLM Integration (Ollama)

**Location:** `backend/src/config/models.py`

**Features:**
- Local LLM inference via Ollama
- Model configuration and presets
- Temperature and sampling parameters
- Support for multiple models (llama3.2, etc.)

**Model Presets:**
- `creative` - High temperature (0.9), creative responses
- `balanced` - Medium temperature (0.7), default
- `precise` - Low temperature (0.3), focused responses
- `deterministic` - Zero temperature (0.0), reproducible

---

## Data Flow

### Chat Request Flow

```
1. User Input
   └─> Frontend (index.html)
       └─> AJAX POST to /api/chat

2. API Proxy
   └─> Flask app.py
       └─> Forward to backend API

3. Backend Processing
   └─> FastAPI api.py
       ├─> Validate request
       ├─> Generate/get thread_id
       └─> Invoke workflow

4. Workflow Execution
   └─> LangGraph workflow.py
       ├─> retrieve node
       │   └─> ChromaDB search
       │       └─> Return relevant docs
       └─> generate node
           └─> Ollama LLM
               └─> Generate response

5. Response Flow
   └─> Backend returns ChatResponse
       └─> Flask forwards to frontend
           └─> JavaScript updates UI
```

---

## Technology Stack

### Frontend
- **Framework:** Flask 3.x
- **Templates:** Jinja2
- **Styling:** Custom CSS
- **JavaScript:** Vanilla JS
- **Deployment:** Vercel

### Backend
- **API:** FastAPI
- **Workflow:** LangGraph
- **LLM Framework:** LangChain
- **Server:** Uvicorn (ASGI)
- **Deployment:** Railway (planned)

### Data & AI
- **Vector DB:** ChromaDB
- **LLM:** Ollama (local)
- **Embeddings:** Default ChromaDB embeddings
- **Models:** llama3.2, others

### DevOps
- **Environment:** Python 3.11+
- **Package Manager:** pip
- **Config:** .env files
- **Logging:** Python logging module

---

## Deployment Architecture

### Development (Local)
```
Frontend:  http://localhost:5000
Backend:   http://localhost:8000
ChromaDB:  Local filesystem
Ollama:    http://localhost:11434
```

### Production
```
Frontend:  Vercel (https://summarizer-agent-langgraph-ufot.vercel.app)
Backend:   Railway (planned)
ChromaDB:  Persistent volume
Ollama:    Railway service (planned)
```

---

## Key Features

### 1. Conversational Memory
- Thread-based conversation tracking
- MemorySaver checkpointer
- Persistent chat history per user

### 2. RAG (Retrieval-Augmented Generation)
- Document retrieval before generation
- Context-aware responses
- Reduces hallucination

### 3. Scalable Architecture
- Stateless API design
- Horizontal scaling ready
- Async workflow execution

### 4. Modular Design
- Separate frontend/backend
- Pluggable components
- Easy to extend/modify

---

## Security & Configuration

### Environment Variables
- `API_BASE_URL` - Backend API endpoint
- `OLLAMA_BASE_URL` - Ollama server URL
- `SECRET_KEY` - Flask session secret
- `DEFAULT_MODEL` - Default LLM model

### CORS Configuration
- Whitelist for Vercel domains
- Localhost for development
- Credentials support enabled

---

## Future Enhancements

### Planned Features
- [ ] User authentication
- [ ] Multi-user support
- [ ] Document upload interface
- [ ] Advanced analytics
- [ ] Rate limiting
- [ ] Caching layer
- [ ] WebSocket streaming
- [ ] Database persistence for chat history

### Scalability
- [ ] Redis for session management
- [ ] PostgreSQL for metadata
- [ ] Container orchestration (K8s)
- [ ] Load balancing
- [ ] CDN for static assets

---

## Development Guidelines

### Adding New Nodes
1. Define node function in `nodes.py`
2. Update `ChatState` if new fields needed
3. Add node to workflow in `workflow.py`
4. Connect edges appropriately

### Adding New Endpoints
1. Define route in `api.py`
2. Create Pydantic models for validation
3. Add CORS origin if needed
4. Update frontend proxy if required

### Data Ingestion
1. Place documents in `backend/data/`
2. Run ingestion script
3. Verify embeddings in ChromaDB
4. Test retrieval quality

---

## Troubleshooting

### Common Issues

**Backend Connection Failed**
- Check if backend is running on port 8000
- Verify API_BASE_URL in frontend
- Check CORS configuration

**Ollama Not Found**
- Ensure Ollama is installed and running
- Pull required model: `ollama pull llama3.2`
- Check OLLAMA_BASE_URL setting

**ChromaDB Errors**
- Verify database directory exists
- Check file permissions
- Re-run setup_chromadb.py if needed

---

## Monitoring & Logging

### Logging Levels
- **INFO** - Normal operations
- **WARNING** - Recoverable issues
- **ERROR** - Failures requiring attention

### Health Checks
- Frontend: `/health` - Returns frontend and backend status
- Backend: `/health` - Returns API operational status

### Metrics (Planned)
- Request/response times
- Error rates
- Model inference time
- ChromaDB query performance

---

## Documentation

### Project Docs
- `README.md` - General overview and quick start
- `ARCHITECTURE.md` - This file (high-level architecture)
- `backend/ARCHITECTURE.md` - Detailed workflow architecture
- `DEPLOYMENT.md` - Deployment instructions
- Various setup guides and troubleshooting docs

---

## Contact & Support

For questions or issues:
1. Check documentation in project root
2. Review backend-specific docs in `backend/`
3. Check troubleshooting guides (OLLAMA_FIX.md, etc.)

---

**Version:** 1.0.0  
**Status:** Active Development  
**License:** MIT (if applicable)
