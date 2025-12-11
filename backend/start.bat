@echo off
REM Quick Start Script for Backend API
REM This script sets up and starts the backend FastAPI server

echo ========================================
echo   Text Analysis Backend - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate
echo.

REM Install/Update dependencies using UV
echo Installing dependencies with UV...
pip install uv
uv pip install -e .
echo.

REM Unzip ChromaDB if needed
if exist "data\chroma_db.zip" (
    if not exist "data\chroma_db" (
        echo Extracting ChromaDB...
        python -c "import zipfile; zipfile.ZipFile('data/chroma_db.zip', 'r').extractall('data')"
        echo ChromaDB extracted successfully.
        echo.
    )
)

REM Check if Ollama is running
echo Checking Ollama...
curl -s http://localhost:11434/api/tags > nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Ollama doesn't seem to be running!
    echo Please start Ollama in another terminal:
    echo   ollama serve
    echo.
    echo Then pull a model:
    echo   ollama pull qwen2.5-coder:0.5b
    echo.
    pause
)

REM Start the API using the script entry point
echo.
echo ========================================
echo   Starting Backend API...
echo   URL: http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.
python api.py
